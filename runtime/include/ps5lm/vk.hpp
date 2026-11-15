// vk.hpp — minimal Vulkan compute layer for the PS5 (RADV GFX1013) and the PC sim.
//
// Design (research/design-history/PROTOTYPE-PLAN.md): no descriptor sets. Every kernel takes
// buffer device addresses (BDA) in push constants, so binding an expert slot
// is a 64-bit pointer write into a small table. Memory:
//   - Buffer::create: driver allocation (host-visible + coherent preferred on UMA,
//     persistently mapped) with SHADER_DEVICE_ADDRESS.
//   - Buffer::import_host: VK_EXT_external_memory_host import of an existing
//     (4 KiB aligned) host mapping, e.g. the io_uring-registered expert slab,
//     so NVMe DMA and GPU reads hit the same bytes (zero-copy).
// One compute queue; submits are synchronous helpers or fence-tracked.
#pragma once

#include <vulkan/vulkan.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps5lm::vk {

inline void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: VkResult " + std::to_string(int(r)));
}

struct Ctx;

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress addr = 0;
    void* map = nullptr;      // persistent host mapping (null if not host-visible)
    uint32_t mem_type = 0;
    bool imported = false;
    template <class T> T* as() const { return static_cast<T*>(map); }
};

struct Pipeline {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t push_bytes = 0;
    std::string name;
};

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    bool has_sparse = false;  // S20
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memprops{};
    double ts_period_ns = 1.0;
    bool has_host_import = false;
    VkDeviceSize host_import_align = 4096;
    PFN_vkGetMemoryHostPointerPropertiesEXT pGetHostPtrProps = nullptr;
    bool has_dmabuf_import = false;
    bool has_coherent = false, has_clock = false;  // S17
    PFN_vkGetMemoryFdPropertiesKHR pGetFdProps = nullptr;

    Ctx() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "ps5lm";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        if (!n) throw std::runtime_error("no Vulkan device");
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(inst, &n, devs.data());
        phys = devs[0];
        for (auto d : devs) {  // prefer a real GPU over llvmpipe
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys = d;
                break;
            }
        }
        vkGetPhysicalDeviceProperties(phys, &props);
        vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
        ts_period_ns = props.limits.timestampPeriod;

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
        qfam = UINT32_MAX;
        for (uint32_t i = 0; i < nq; ++i)  // the graphics+compute family: on PS5 RADV everything rides one ring anyway
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) break; }
        if (qfam == UINT32_MAX) throw std::runtime_error("no compute queue");

        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> exts(ne);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, exts.data());
        std::vector<const char*> want;
        for (auto& e : exts)
            if (!std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) has_host_import = true;
        if (has_host_import) want.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        bool fd_ext = false, dmabuf_ext = false;
        for (auto& e : exts) {
            if (!std::strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) fd_ext = true;
            if (!std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)) dmabuf_ext = true;
        }
        has_dmabuf_import = fd_ext && dmabuf_ext;
        for (auto& e : exts) {  // S17: GPU-uncached control memory + realtime clock for the one-cb-per-token gates
            if (!std::strcmp(e.extensionName, "VK_AMD_device_coherent_memory")) has_coherent = true;
            if (!std::strcmp(e.extensionName, VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) has_clock = true;
        }
        if (has_coherent) want.push_back("VK_AMD_device_coherent_memory");
        if (has_clock) want.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
        if (has_dmabuf_import) {
            want.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            want.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
        }

        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.subgroupSizeControl = VK_TRUE;
        f13.computeFullSubgroups = VK_TRUE;
        f13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.pNext = &f13;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.timelineSemaphore = VK_TRUE;
        f12.shaderFloat16 = VK_TRUE;
        f12.shaderInt8 = VK_TRUE;
        f12.storageBuffer8BitAccess = VK_TRUE;
        f12.scalarBlockLayout = VK_TRUE;
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.pNext = &f12;
        f11.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceCoherentMemoryFeaturesAMD fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD};
        fcm.deviceCoherentMemory = VK_TRUE;
        VkPhysicalDeviceShaderClockFeaturesKHR fclk{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
        fclk.shaderDeviceClock = VK_TRUE;
        fclk.shaderSubgroupClock = VK_TRUE;
        void* chain = &f11;
        if (has_coherent) { fcm.pNext = chain; chain = &fcm; }
        if (has_clock) { fclk.pNext = chain; chain = &fclk; }
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = chain;
        f2.features.shaderInt64 = VK_TRUE;
        f2.features.shaderInt16 = VK_TRUE;
        {   // S20 elastic KV: sparse (partially resident) buffers, bound on the compute queue
            VkPhysicalDeviceFeatures sup{};
            vkGetPhysicalDeviceFeatures(phys, &sup);
            uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qp(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp.data());
            has_sparse = sup.sparseBinding && sup.sparseResidencyBuffer && qfam < nq && (qp[qfam].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
            if (has_sparse) { f2.features.sparseBinding = VK_TRUE; f2.features.sparseResidencyBuffer = VK_TRUE; }
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = uint32_t(want.size());
        dci.ppEnabledExtensionNames = want.data();
        check(vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice");
        vkGetDeviceQueue(dev, qfam, 0, &queue);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam;
        check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");
        if (has_host_import) {
            pGetHostPtrProps = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
                vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT"));
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &hp;
            vkGetPhysicalDeviceProperties2(phys, &p2);
            host_import_align = hp.minImportedHostPointerAlignment;
        }
        if (has_dmabuf_import)
            pGetFdProps = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(dev, "vkGetMemoryFdPropertiesKHR"));
    }
    ~Ctx() {
        if (dev) {
            vkDeviceWaitIdle(dev);
            vkDestroyCommandPool(dev, pool, nullptr);
            vkDestroyDevice(dev, nullptr);
        }
        if (inst) vkDestroyInstance(inst, nullptr);
    }
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;

    void print_memory() const {
        for (uint32_t h = 0; h < memprops.memoryHeapCount; ++h)
            std::fprintf(stderr, "heap %u: %.2f GiB%s\n", h, double(memprops.memoryHeaps[h].size) / (1ull << 30),
                         memprops.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "");
        for (uint32_t t = 0; t < memprops.memoryTypeCount; ++t) {
            auto f = memprops.memoryTypes[t].propertyFlags;
            std::fprintf(stderr, "  type %u heap %u:%s%s%s%s\n", t, memprops.memoryTypes[t].heapIndex,
                         f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                         f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HOST_VISIBLE" : "",
                         f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " COHERENT" : "",
                         f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " CACHED" : "");
        }
    }

    // Choose a memory type: all `need` flags; prefer `want`; avoid `avoid`.
    uint32_t pick_type(uint32_t bits, VkMemoryPropertyFlags need, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0) const {
        int best = -1, best_score = -1;
        for (uint32_t t = 0; t < memprops.memoryTypeCount; ++t) {
            if (!(bits & (1u << t))) continue;
            auto f = memprops.memoryTypes[t].propertyFlags;
            if ((f & need) != need) continue;
            int score = __builtin_popcount(f & want) * 4 - __builtin_popcount(f & avoid) * 2;
            score = score * 64 + int(double(memprops.memoryHeaps[memprops.memoryTypes[t].heapIndex].size) / double(1ull << 30));
            if (score > best_score) { best_score = score; best = int(t); }
        }
        if (best < 0) throw std::runtime_error("no suitable memory type");
        return uint32_t(best);
    }

    static constexpr VkBufferUsageFlags kUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // host_visible=true: HOST_VISIBLE|COHERENT (persistently mapped), DEVICE_LOCAL preferred.
    // host_visible=false: DEVICE_LOCAL only (fastest GPU path; on UMA = the VRAM carve).
    Buffer create(VkDeviceSize size, bool host_visible = true, VkMemoryPropertyFlags want_extra = 0, VkMemoryPropertyFlags avoid = 0) {
        Buffer b;
        b.size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = kUsage;
        check(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b.buf, &mr);
        VkMemoryPropertyFlags need = host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                  : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        b.mem_type = pick_type(mr.memoryTypeBits, need, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | want_extra, avoid);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &fi;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = b.mem_type;
        check(vkAllocateMemory(dev, &mai, nullptr, &b.mem), "vkAllocateMemory");
        check(vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory");
        if (memprops.memoryTypes[b.mem_type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            check(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
        finish(b);
        return b;
    }

    // Import host memory (must be host_import_align aligned, size multiple of it).
    Buffer import_host(void* ptr, VkDeviceSize size) {
        if (!has_host_import) throw std::runtime_error("VK_EXT_external_memory_host unavailable");
        Buffer b;
        b.size = size;
        b.imported = true;
        b.map = ptr;
        VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.pNext = &ebi;
        bci.size = size;
        bci.usage = kUsage;
        check(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer(import)");
        VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        check(pGetHostPtrProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hp), "vkGetMemoryHostPointerPropertiesEXT");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b.buf, &mr);
        b.mem_type = pick_type(mr.memoryTypeBits & hp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = ptr;
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.pNext = &imp;
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &fi;
        mai.allocationSize = size;
        mai.memoryTypeIndex = b.mem_type;
        check(vkAllocateMemory(dev, &mai, nullptr, &b.mem), "vkAllocateMemory(import host ptr)");
        check(vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory(import)");
        finish(b);
        return b;
    }

    // Import a dma-buf (e.g. a udmabuf over a memfd). Unlike a host-pointer (userptr) import, the
    // kernel pins the pages once at import: no per-submission page-table walk (measured S13: userptr
    // imports cost ~3 ms per GiB per vkQueueSubmit on amdgpu because every BDA-visible BO rides
    // every submit). Vulkan takes ownership of a dup of `fd`.
    Buffer import_dmabuf(int fd, VkDeviceSize size) {
        if (!has_dmabuf_import) throw std::runtime_error("dma-buf import unavailable");
        Buffer b;
        b.size = size;
        b.imported = true;
        VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.pNext = &ebi;
        bci.size = size;
        bci.usage = kUsage;
        check(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer(dmabuf)");
        VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        check(pGetFdProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp), "vkGetMemoryFdPropertiesKHR");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b.buf, &mr);
        uint32_t bits = mr.memoryTypeBits & fp.memoryTypeBits;
        if (!bits) bits = fp.memoryTypeBits;
        b.mem_type = pick_type(bits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        imp.fd = ::dup(fd);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.pNext = &imp;
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &fi;
        mai.allocationSize = size;
        mai.memoryTypeIndex = b.mem_type;
        check(vkAllocateMemory(dev, &mai, nullptr, &b.mem), "vkAllocateMemory(import dma-buf)");
        check(vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory(dmabuf)");
        finish(b);
        return b;
    }

    void destroy(Buffer& b) {
        if (!b.buf) return;
        if (b.map && !b.imported) vkUnmapMemory(dev, b.mem);
        vkDestroyBuffer(dev, b.buf, nullptr);
        vkFreeMemory(dev, b.mem, nullptr);
        b = {};
    }

    Pipeline pipeline(const std::string& spv_path, uint32_t push_bytes, uint32_t subgroup = 64) {
        std::ifstream f(spv_path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open shader " + spv_path);
        std::vector<char> code(size_t(f.tellg()));
        f.seekg(0);
        f.read(code.data(), std::streamsize(code.size()));
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size();
        smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule mod;
        check(vkCreateShaderModule(dev, &smi, nullptr, &mod), "vkCreateShaderModule");
        Pipeline p;
        p.name = spv_path;
        p.push_bytes = push_bytes;
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.pushConstantRangeCount = push_bytes ? 1 : 0;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(dev, &pli, nullptr, &p.layout), "vkCreatePipelineLayout");
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        rss.requiredSubgroupSize = subgroup;
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.pNext = subgroup ? &rss : nullptr;
        cpi.stage.flags = subgroup ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = p.layout;
        check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe), "vkCreateComputePipelines");
        vkDestroyShaderModule(dev, mod, nullptr);
        return p;
    }
    void destroy(Pipeline& p) {
        if (p.pipe) vkDestroyPipeline(dev, p.pipe, nullptr);
        if (p.layout) vkDestroyPipelineLayout(dev, p.layout, nullptr);
        p = {};
    }

    VkCommandBuffer cmd() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        check(vkAllocateCommandBuffers(dev, &ai, &cb), "vkAllocateCommandBuffers");
        return cb;
    }
    static void begin(VkCommandBuffer cb, bool one_time = true) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = one_time ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
        check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    }
    static void end(VkCommandBuffer cb) { check(vkEndCommandBuffer(cb), "vkEndCommandBuffer"); }

    static void dispatch(VkCommandBuffer cb, const Pipeline& p, const void* push, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        if (p.push_bytes) vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.push_bytes, push);
        vkCmdDispatch(cb, gx, gy, gz);
    }
    // Compute -> compute (and host) visibility.
    // S17: g_barrier_host = false -> compute -> compute only (the host reads results after the submit's fence, whose
    // end-of-pipe release writes the GPU L2 back anyway; the one-cb mailboxes live in GPU-uncached memory).
    static inline bool g_barrier_host = true;
    static void barrier(VkCommandBuffer cb) {
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (g_barrier_host ? VK_PIPELINE_STAGE_2_HOST_BIT : 0);
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | (g_barrier_host ? VK_ACCESS_2_HOST_READ_BIT : 0);
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    VkFence fence() {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence f;
        check(vkCreateFence(dev, &fci, nullptr, &f), "vkCreateFence");
        return f;
    }
    void submit(VkCommandBuffer cb, VkFence f) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        check(vkQueueSubmit(queue, 1, &si, f), "vkQueueSubmit");
    }
    void wait(VkFence f) {
        check(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        vkResetFences(dev, 1, &f);
    }
    // S15: busy-poll a fence (no sleep/wake: the blocking wait costs ~tens of us of scheduler latency per layer-step,
    // 49 times per token). For a dedicated, pinned host thread only.
    void wait_spin(VkFence f) {
        VkResult r;
        while ((r = vkGetFenceStatus(dev, f)) == VK_NOT_READY) __builtin_ia32_pause();
        check(r, "vkGetFenceStatus");
        vkResetFences(dev, 1, &f);
    }
    void run(VkCommandBuffer cb) {
        VkFence f = fence();
        submit(cb, f);
        wait(f);
        vkDestroyFence(dev, f, nullptr);
    }

    // S17: host-visible, GPU-UNCACHED memory (VK_AMD_device_coherent_memory): GPU loads/stores go to DRAM, so a shader can
    // poll a word the host writes mid-command-buffer and the host sees GPU writes without a fence. Throws if unavailable.
    Buffer create_uncached(VkDeviceSize size) {
        if (!has_coherent) throw std::runtime_error("VK_AMD_device_coherent_memory unavailable");
        constexpr VkMemoryPropertyFlags kUnc = VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
        Buffer b = create(size, true, kUnc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if ((memprops.memoryTypes[b.mem_type].propertyFlags & kUnc) != kUnc) { destroy(b); throw std::runtime_error("no GPU-uncached host-visible memory type"); }
        return b;
    }

    VkQueryPool timestamps(uint32_t n) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = n;
        VkQueryPool qp;
        check(vkCreateQueryPool(dev, &qi, nullptr, &qp), "vkCreateQueryPool");
        return qp;
    }
    std::vector<uint64_t> read_ts(VkQueryPool qp, uint32_t n) {
        std::vector<uint64_t> v(n);
        check(vkGetQueryPoolResults(dev, qp, 0, n, n * 8, v.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults");
        return v;
    }

private:
public:
    // S20: a partially resident buffer of `size` bytes (virtual range only); alignment / memory types for its binds
    Buffer create_sparse(VkDeviceSize size, VkDeviceSize& align, uint32_t& type_bits) {
        if (!has_sparse) throw std::runtime_error("sparse buffers unavailable");
        Buffer b;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
        bci.size = size;
        bci.usage = kUsage;
        check(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer(sparse)");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b.buf, &mr);
        align = mr.alignment;
        type_bits = mr.memoryTypeBits;
        b.size = mr.size;
        finish(b);
        return b;
    }
    // Device memory for sparse binds (GTT unless asked otherwise), device-address capable
    VkDeviceMemory alloc_raw(VkDeviceSize size, uint32_t type_bits, uint32_t& type) {
        type = pick_type(type_bits, 0, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &fi;
        mai.allocationSize = size;
        mai.memoryTypeIndex = type;
        VkDeviceMemory m = VK_NULL_HANDLE;
        check(vkAllocateMemory(dev, &mai, nullptr, &m), "vkAllocateMemory(raw)");
        return m;
    }
    // One vkQueueBindSparse for several buffers, waited on (the queue is idle between tokens when the engine calls it)
    void bind_sparse(const std::vector<VkSparseBufferMemoryBindInfo>& bi) {
        if (bi.empty()) return;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence f = VK_NULL_HANDLE;
        check(vkCreateFence(dev, &fci, nullptr, &f), "vkCreateFence(sparse)");
        VkBindSparseInfo bs{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        bs.bufferBindCount = uint32_t(bi.size());
        bs.pBufferBinds = bi.data();
        check(vkQueueBindSparse(queue, 1, &bs, f), "vkQueueBindSparse");
        check(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX), "vkWaitForFences(sparse)");
        vkDestroyFence(dev, f, nullptr);
    }
private:
    void finish(Buffer& b) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b.buf;
        b.addr = vkGetBufferDeviceAddress(dev, &ai);
    }
};

}  // namespace ps5lm::vk

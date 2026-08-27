//! HAR-owned Vulkan resource and submission boundary.
//!
//! The public API exposes only HAR handles and byte-oriented operations.  It
//! deliberately has no foreign inference dependency and performs no CPU
//! fallback or implicit upload. GPU work is supplied as SPIR-V words and executed by
//! Vulkan compute pipelines.

use ash::{vk, Entry, Instance};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};

pub type HarResult<T> = Result<T, HarError>;

#[derive(Debug, Clone)]
pub struct HarError {
    pub operation: &'static str,
    pub message: String,
    pub result: Option<vk::Result>,
}

impl HarError {
    fn argument(operation: &'static str, message: impl Into<String>) -> Self {
        Self {
            operation,
            message: message.into(),
            result: None,
        }
    }
    fn vulkan(operation: &'static str, result: vk::Result) -> Self {
        Self {
            operation,
            message: format!("{operation} failed: {result:?}"),
            result: Some(result),
        }
    }
    fn load(message: impl Into<String>) -> Self {
        Self {
            operation: "load Vulkan",
            message: message.into(),
            result: None,
        }
    }
}

impl Display for HarError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.operation, self.message)
    }
}
impl std::error::Error for HarError {}

fn vk_result<T>(operation: &'static str, value: ash::prelude::VkResult<T>) -> HarResult<T> {
    value.map_err(|result| HarError::vulkan(operation, result))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPreference {
    DeviceLocal,
    HostVisible,
    HostVisibleDeviceLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    Compute,
    Transfer,
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceOptions {
    pub device_index: Option<usize>,
    pub prefer_discrete: bool,
    pub require_wave32: bool,
}
impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            device_index: None,
            prefer_discrete: true,
            require_wave32: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueFamilyInfo {
    pub index: u32,
    pub queue_count: u32,
    pub flags: vk::QueueFlags,
    pub timestamp_valid_bits: u32,
}

#[derive(Debug, Clone)]
pub struct MemoryTypeInfo {
    pub index: u32,
    pub heap_index: u32,
    pub properties: vk::MemoryPropertyFlags,
}

#[derive(Debug, Clone)]
pub struct Capabilities {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub wave32_supported: bool,
    pub wave64_supported: bool,
    pub integer_dot_product: bool,
    pub shader_float16: bool,
    pub bf16_extension_present: bool,
    pub cooperative_matrix_extension_present: bool,
    pub cooperative_matrix: bool,
    pub timeline_semaphore: bool,
    pub synchronization2: bool,
    pub subgroup_size_control: bool,
    pub timestamp_period_ns: f32,
    pub timestamp_compute_and_graphics: bool,
    pub max_bound_descriptor_sets: u32,
    pub max_push_constants_size: u32,
    pub max_storage_buffer_range: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub non_coherent_atom_size: u64,
    pub max_compute_workgroup_invocations: u32,
    pub max_compute_workgroup_size: [u32; 3],
    pub heaps: Vec<vk::MemoryHeap>,
    pub memory_types: Vec<MemoryTypeInfo>,
    pub queue_families: Vec<QueueFamilyInfo>,
}

impl Capabilities {
    pub fn timestamp_queries(&self) -> bool {
        self.timestamp_period_ns > 0.0
            && self
                .queue_families
                .iter()
                .any(|q| q.timestamp_valid_bits != 0)
    }

    pub fn to_json(&self) -> String {
        // Keep this dependency-free: the phenotype artifact can also be
        // imported from an external probe, and this representation is for Rust
        // callers.
        let heaps = self
            .heaps
            .iter()
            .enumerate()
            .map(|(i, h)| {
                format!(
                    "{{\"index\":{i},\"size\":{},\"flags\":{}}}",
                    h.size,
                    h.flags.as_raw()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let types = self
            .memory_types
            .iter()
            .map(|t| {
                format!(
                    "{{\"index\":{},\"heap_index\":{},\"properties\":{}}}",
                    t.index,
                    t.heap_index,
                    t.properties.as_raw()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let queues =
            self.queue_families
                .iter()
                .map(|q| {
                    format!(
            "{{\"index\":{},\"queue_count\":{},\"flags\":{},\"timestamp_valid_bits\":{}}}",
            q.index, q.queue_count, q.flags.as_raw(), q.timestamp_valid_bits
        )
                })
                .collect::<Vec<_>>()
                .join(",");
        format!(
            "{{\"schema\":\"har.rdna4.rust_capabilities.v1\",\"device\":{{\"name\":{:?},\"vendor_id\":{},\"device_id\":{},\"api_version\":{},\"driver_version\":{}}},\"subgroup\":{{\"reported_size\":{},\"min_size\":{},\"max_size\":{},\"wave32_supported\":{},\"wave64_supported\":{}}},\"arithmetic\":{{\"integer_dot_product\":{},\"fp16_type\":{},\"bf16_extension_present\":{},\"cooperative_matrix\":{}}},\"execution\":{{\"timeline_semaphore\":{},\"synchronization2\":{},\"subgroup_size_control\":{},\"timestamp_period_ns\":{}}},\"limits\":{{\"max_bound_descriptor_sets\":{},\"max_push_constants_size\":{},\"max_storage_buffer_range\":{},\"min_storage_buffer_offset_alignment\":{},\"non_coherent_atom_size\":{},\"max_compute_workgroup_invocations\":{},\"max_compute_workgroup_size\":[{},{},{}]}},\"heaps\":[{}],\"memory_types\":[{}],\"queue_families\":[{}]}}",
            self.name, self.vendor_id, self.device_id, self.api_version, self.driver_version,
            self.subgroup_size, self.min_subgroup_size, self.max_subgroup_size,
            self.wave32_supported, self.wave64_supported, self.integer_dot_product,
            self.shader_float16, self.bf16_extension_present, self.cooperative_matrix,
            self.timeline_semaphore, self.synchronization2,
            self.subgroup_size_control, self.timestamp_period_ns,
            self.max_bound_descriptor_sets, self.max_push_constants_size, self.max_storage_buffer_range,
            self.min_storage_buffer_offset_alignment, self.non_coherent_atom_size,
            self.max_compute_workgroup_invocations, self.max_compute_workgroup_size[0],
            self.max_compute_workgroup_size[1], self.max_compute_workgroup_size[2], heaps, types, queues
        )
    }
}

struct DeviceInner {
    _entry: Entry,
    instance: Instance,
    device: ash::Device,
    capabilities: Capabilities,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    compute_family: u32,
    transfer_family: u32,
    compute_queue: vk::Queue,
    transfer_queue: vk::Queue,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // SAFETY: DeviceInner owns the logical device and all child wrappers
        // hold an Arc<DeviceInner>, so children are dropped before this point.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Owns an instance, selected physical device, and logical device.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    pub fn open(options: DeviceOptions) -> HarResult<Self> {
        // SAFETY: Entry::load only resolves Vulkan loader symbols; the returned
        // Entry owns no borrowed application memory.
        let entry = unsafe { Entry::load().map_err(|e| HarError::load(e.to_string()))? };
        let app_name = CString::new("HAR RDNA4 native Rust backend").expect("literal has no NUL");
        let engine_name = CString::new("HAR").expect("literal has no NUL");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: app_info and instance_info point to stack values that remain
        // alive for the call; no extensions/layers are requested implicitly.
        let instance = unsafe {
            vk_result(
                "create instance",
                entry.create_instance(&instance_info, None),
            )?
        };
        // SAFETY: instance is owned above and valid until DeviceInner drop.
        let physicals = unsafe {
            vk_result(
                "enumerate physical devices",
                instance.enumerate_physical_devices(),
            )?
        };
        if physicals.is_empty() {
            // SAFETY: no child object exists and instance is exclusively owned here.
            unsafe {
                instance.destroy_instance(None);
            }
            return Err(HarError::argument(
                "select device",
                "no Vulkan physical device",
            ));
        }

        let mut selected = None;
        let mut selected_score = i32::MIN;
        for (index, physical) in physicals.iter().copied().enumerate() {
            if options.device_index.is_some_and(|wanted| wanted != index) {
                continue;
            }
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            // SAFETY: props2's p_next points to a correctly typed, live output
            // struct and Vulkan writes only the documented chain fields.
            unsafe {
                instance.get_physical_device_properties2(physical, &mut props2);
            }
            let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut props_chain = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut v12)
                .push_next(&mut v13);
            // SAFETY: the pNext chain is composed of Vulkan output structs with
            // valid sType values and lives across this call.
            unsafe {
                instance.get_physical_device_properties2(physical, &mut props_chain);
            }
            let wave32 = v13.min_subgroup_size <= 32 && v13.max_subgroup_size >= 32;
            if options.require_wave32 && !wave32 {
                continue;
            }
            let mut score = if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                100
            } else {
                0
            };
            score += if wave32 { 20 } else { 0 };
            if options.prefer_discrete
                && properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU
            {
                score -= 1000;
            }
            if score > selected_score {
                selected_score = score;
                selected = Some((physical, subgroup, v12, v13));
            }
        }
        let (physical, subgroup, _v12_properties, v13_properties) = selected.ok_or_else(|| {
            HarError::argument(
                "select device",
                "no physical device satisfies HAR Wave32 requirements",
            )
        })?;

        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let compute_family = families
            .iter()
            .position(|q| {
                q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                families
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
            .ok_or_else(|| HarError::argument("select queue", "no compute queue family"))?
            as u32;
        let transfer_family = families
            .iter()
            .position(|q| {
                q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                families
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER))
            })
            .ok_or_else(|| HarError::argument("select queue", "no transfer queue family"))?
            as u32;
        let mut unique = vec![compute_family];
        if transfer_family != compute_family {
            unique.push(transfer_family);
        }
        let priorities = vec![1.0_f32; unique.len()];
        let queue_infos = unique
            .iter()
            .enumerate()
            .map(|(i, family)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*family)
                    .queue_priorities(std::slice::from_ref(&priorities[i]))
            })
            .collect::<Vec<_>>();

        let bf16_extension_name =
            CString::new("VK_KHR_shader_bfloat16").expect("literal has no NUL");
        let has_bf16_extension =
            device_extension_present(&instance, physical, bf16_extension_name.as_c_str());
        let has_cooperative_matrix_extension =
            device_extension_present(&instance, physical, vk::KHR_COOPERATIVE_MATRIX_NAME);
        let mut available_v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut available_v13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut available_cooperative_matrix =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut available_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut available_v12)
            .push_next(&mut available_v13);
        if has_cooperative_matrix_extension {
            available_features = available_features.push_next(&mut available_cooperative_matrix);
        }
        // SAFETY: output feature chain is valid and physical is selected from
        // enumerate_physical_devices above.
        unsafe {
            instance.get_physical_device_features2(physical, &mut available_features);
        }
        if available_v12.timeline_semaphore != vk::TRUE
            || available_v13.synchronization2 != vk::TRUE
        {
            unsafe {
                instance.destroy_instance(None);
            }
            return Err(HarError::argument(
                "create device",
                "timeline semaphore and synchronization2 are required",
            ));
        }
        if options.require_wave32 && available_v13.subgroup_size_control != vk::TRUE {
            unsafe {
                instance.destroy_instance(None);
            }
            return Err(HarError::argument(
                "create device",
                "subgroup size control is required for Wave32",
            ));
        }
        let mut enabled_v12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(available_v12.buffer_device_address == vk::TRUE)
            .shader_float16(available_v12.shader_float16 == vk::TRUE)
            .shader_int8(available_v12.shader_int8 == vk::TRUE);
        let mut enabled_v13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .subgroup_size_control(available_v13.subgroup_size_control == vk::TRUE)
            .compute_full_subgroups(available_v13.compute_full_subgroups == vk::TRUE)
            .shader_integer_dot_product(available_v13.shader_integer_dot_product == vk::TRUE)
            .maintenance4(available_v13.maintenance4 == vk::TRUE);
        let mut enabled_cooperative_matrix =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut enabled_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enabled_v12)
            .push_next(&mut enabled_v13);
        if has_cooperative_matrix_extension
            && available_cooperative_matrix.cooperative_matrix == vk::TRUE
        {
            enabled_cooperative_matrix = enabled_cooperative_matrix
                .cooperative_matrix(true)
                .cooperative_matrix_robust_buffer_access(
                    available_cooperative_matrix.cooperative_matrix_robust_buffer_access
                        == vk::TRUE,
                );
            enabled_features = enabled_features.push_next(&mut enabled_cooperative_matrix);
        }
        let mut enabled_extensions = Vec::new();
        if has_cooperative_matrix_extension {
            enabled_extensions.push(vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr());
        }
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions)
            .push_next(&mut enabled_features);
        // SAFETY: queue_infos and enabled feature chains remain live during the
        // call; physical is valid and no unsupported extension is enabled.
        let device = match unsafe { instance.create_device(physical, &device_info, None) } {
            Ok(device) => device,
            Err(error) => {
                unsafe {
                    instance.destroy_instance(None);
                }
                return Err(HarError::vulkan("create device", error));
            }
        };
        // SAFETY: queue family/index were included in device_info.
        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };
        // SAFETY: queue family/index were included in device_info.
        let transfer_queue = unsafe { device.get_device_queue(transfer_family, 0) };
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let capabilities = Capabilities {
            name: unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            subgroup_size: subgroup.subgroup_size,
            min_subgroup_size: v13_properties.min_subgroup_size,
            max_subgroup_size: v13_properties.max_subgroup_size,
            wave32_supported: v13_properties.min_subgroup_size <= 32
                && v13_properties.max_subgroup_size >= 32,
            wave64_supported: v13_properties.min_subgroup_size <= 64
                && v13_properties.max_subgroup_size >= 64,
            integer_dot_product: available_v13.shader_integer_dot_product == vk::TRUE,
            shader_float16: available_v12.shader_float16 == vk::TRUE,
            bf16_extension_present: has_bf16_extension,
            cooperative_matrix_extension_present: has_cooperative_matrix_extension,
            cooperative_matrix: has_cooperative_matrix_extension
                && available_cooperative_matrix.cooperative_matrix == vk::TRUE,
            timeline_semaphore: available_v12.timeline_semaphore == vk::TRUE,
            synchronization2: available_v13.synchronization2 == vk::TRUE,
            subgroup_size_control: available_v13.subgroup_size_control == vk::TRUE,
            timestamp_period_ns: properties.limits.timestamp_period,
            timestamp_compute_and_graphics: properties.limits.timestamp_compute_and_graphics
                == vk::TRUE,
            max_bound_descriptor_sets: properties.limits.max_bound_descriptor_sets,
            max_push_constants_size: properties.limits.max_push_constants_size,
            max_storage_buffer_range: properties.limits.max_storage_buffer_range as u64,
            min_storage_buffer_offset_alignment: properties
                .limits
                .min_storage_buffer_offset_alignment
                as u64,
            non_coherent_atom_size: properties.limits.non_coherent_atom_size as u64,
            max_compute_workgroup_invocations: properties.limits.max_compute_work_group_invocations,
            max_compute_workgroup_size: properties.limits.max_compute_work_group_size,
            heaps: memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize]
                .to_vec(),
            memory_types: memory_properties.memory_types
                [..memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .map(|(index, t)| MemoryTypeInfo {
                    index: index as u32,
                    heap_index: t.heap_index,
                    properties: t.property_flags,
                })
                .collect(),
            queue_families: families
                .iter()
                .enumerate()
                .map(|(index, q)| QueueFamilyInfo {
                    index: index as u32,
                    queue_count: q.queue_count,
                    flags: q.queue_flags,
                    timestamp_valid_bits: q.timestamp_valid_bits,
                })
                .collect(),
        };
        Ok(Self {
            inner: Arc::new(DeviceInner {
                _entry: entry,
                instance,
                device,
                capabilities,
                memory_properties,
                compute_family,
                transfer_family,
                compute_queue,
                transfer_queue,
            }),
        })
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.inner.capabilities
    }
    pub fn create_buffer(
        &self,
        size: usize,
        usage: vk::BufferUsageFlags,
        preference: MemoryPreference,
        name: &str,
    ) -> HarResult<Buffer> {
        if size == 0 {
            return Err(HarError::argument(
                "create buffer",
                "zero-sized buffers are not valid",
            ));
        }
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: info contains only POD values and points to no temporary data.
        let buffer = vk_result("create buffer", unsafe {
            self.inner.device.create_buffer(&info, None)
        })?;
        // SAFETY: buffer is a live child of the owned device.
        let requirements = unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };
        let memory_type = choose_memory_type(
            &self.inner.memory_properties,
            requirements.memory_type_bits,
            preference,
        )
        .ok_or_else(|| {
            unsafe {
                self.inner.device.destroy_buffer(buffer, None);
            }
            HarError::argument(
                "allocate buffer",
                "memory preference has no compatible type",
            )
        })?;
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: allocation_info uses requirements returned by the same device.
        let memory = match unsafe { self.inner.device.allocate_memory(&allocation_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe {
                    self.inner.device.destroy_buffer(buffer, None);
                }
                return Err(HarError::vulkan("allocate buffer memory", error));
            }
        };
        // SAFETY: offset zero satisfies the memory requirement alignment and
        // buffer/memory were created from this exact logical device.
        if let Err(error) = unsafe { self.inner.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.inner.device.free_memory(memory, None);
                self.inner.device.destroy_buffer(buffer, None);
            }
            return Err(HarError::vulkan("bind buffer memory", error));
        }
        let properties =
            self.inner.memory_properties.memory_types[memory_type as usize].property_flags;
        Ok(Buffer {
            inner: Arc::new(BufferInner {
                device: self.inner.clone(),
                handle: buffer,
                allocation: Allocation {
                    device: self.inner.clone(),
                    memory,
                    size: requirements.size,
                    properties,
                },
                size,
                usage,
                name: name.to_owned(),
                generation: 1,
            }),
        })
    }
    pub fn create_queue(&self, kind: QueueKind) -> HarResult<Queue> {
        let (family, handle) = match kind {
            QueueKind::Compute => (self.inner.compute_family, self.inner.compute_queue),
            QueueKind::Transfer => (self.inner.transfer_family, self.inner.transfer_queue),
        };
        Queue::new(self.inner.clone(), kind, family, handle)
    }
    pub fn create_pipeline_cache(&self, initial_data: &[u8]) -> HarResult<PipelineCache> {
        let info = vk::PipelineCacheCreateInfo::default().initial_data(initial_data);
        // SAFETY: initial_data remains live for the duration of this call.
        let cache = vk_result("create pipeline cache", unsafe {
            self.inner.device.create_pipeline_cache(&info, None)
        })?;
        Ok(PipelineCache {
            device: self.inner.clone(),
            handle: cache,
        })
    }
    pub fn create_timestamp_query(&self, count: u32) -> HarResult<TimestampQuery> {
        if count < 2 || !self.capabilities().timestamp_queries() {
            return Err(HarError::argument(
                "create timestamp query",
                "timestamps unavailable or fewer than two queries requested",
            ));
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        // SAFETY: info is a local POD create description and the device owns the result.
        let pool = vk_result("create query pool", unsafe {
            self.inner.device.create_query_pool(&info, None)
        })?;
        Ok(TimestampQuery {
            inner: Arc::new(TimestampQueryInner {
                device: self.inner.clone(),
                pool,
                count,
            }),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn create_pipeline(
        &self,
        spirv: &[u32],
        bindings: &[(u32, u32)],
        push_constant_bytes: u32,
        local_size_x: u32,
        require_wave32: bool,
        name: &str,
        cache: Option<&PipelineCache>,
    ) -> HarResult<Pipeline> {
        self.create_pipeline_specialized(
            spirv,
            bindings,
            push_constant_bytes,
            local_size_x,
            if require_wave32 { Some(32) } else { None },
            name,
            cache,
            &[],
        )
    }

    /// HARSF S2: pipeline creation with SPIR-V specialization constants.
    /// `spec_constants` maps specialization ids to u32 values (the d32a mmv
    /// family uses id 0 = BLOCK_SIZE/local_x, id 1 = NUM_ROWS,
    /// id 2 = NUM_COLS). HAR owns pipeline selection end to end.
    #[allow(clippy::too_many_arguments)]
    pub fn create_pipeline_specialized(
        &self,
        spirv: &[u32],
        bindings: &[(u32, u32)],
        push_constant_bytes: u32,
        local_size_x: u32,
        required_subgroup: Option<u32>,
        name: &str,
        cache: Option<&PipelineCache>,
        spec_constants: &[(u32, u32)],
    ) -> HarResult<Pipeline> {
        let mut spec_data: Vec<u32> = Vec::with_capacity(spec_constants.len());
        let mut spec_entries: Vec<vk::SpecializationMapEntry> =
            Vec::with_capacity(spec_constants.len());
        for (i, (id, value)) in spec_constants.iter().enumerate() {
            spec_entries.push(
                vk::SpecializationMapEntry::default()
                    .constant_id(*id)
                    .offset((i * 4) as u32)
                    .size(4),
            );
            spec_data.push(*value);
        }
        let mut spec_info = vk::SpecializationInfo::default();
        if !spec_entries.is_empty() {
            spec_info = spec_info.map_entries(&spec_entries).data(unsafe {
                std::slice::from_raw_parts(spec_data.as_ptr().cast(), spec_data.len() * 4)
            });
        }
        self.create_pipeline_with_required_subgroup_inner(
            spirv,
            bindings,
            push_constant_bytes,
            local_size_x,
            required_subgroup,
            name,
            cache,
            if spec_entries.is_empty() {
                None
            } else {
                Some(&spec_info)
            },
        )
    }
    pub fn create_pipeline_wave64(
        &self,
        spirv: &[u32],
        bindings: &[(u32, u32)],
        push_constant_bytes: u32,
        local_size_x: u32,
        name: &str,
        cache: Option<&PipelineCache>,
    ) -> HarResult<Pipeline> {
        self.create_pipeline_specialized(
            spirv,
            bindings,
            push_constant_bytes,
            local_size_x,
            Some(64),
            name,
            cache,
            &[],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn create_pipeline_with_required_subgroup_inner(
        &self,
        spirv: &[u32],
        bindings: &[(u32, u32)],
        push_constant_bytes: u32,
        local_size_x: u32,
        required_subgroup: Option<u32>,
        name: &str,
        cache: Option<&PipelineCache>,
        spec_info: Option<&vk::SpecializationInfo>,
    ) -> HarResult<Pipeline> {
        if spirv.is_empty() || spirv[0] != 0x0723_0203 {
            return Err(HarError::argument(
                "create pipeline",
                "SPIR-V magic is missing",
            ));
        }
        if push_constant_bytes > self.capabilities().max_push_constants_size {
            return Err(HarError::argument(
                "create pipeline",
                "push constants exceed device limit",
            ));
        }
        if local_size_x == 0 || local_size_x > self.capabilities().max_compute_workgroup_invocations
        {
            return Err(HarError::argument(
                "create pipeline",
                "local size exceeds device limit",
            ));
        }
        if let Some(required) = required_subgroup {
            let available = match required {
                32 => self.capabilities().wave32_supported,
                64 => self.capabilities().wave64_supported,
                _ => false,
            };
            if !available {
                return Err(HarError::argument(
                    "create pipeline",
                    "requested subgroup size is unavailable",
                ));
            }
        }
        let mut declarations = Vec::with_capacity(bindings.len());
        let mut sorted = BTreeMap::new();
        for (binding, count) in bindings {
            if *count == 0 || sorted.insert(*binding, *count).is_some() {
                return Err(HarError::argument(
                    "create pipeline",
                    "duplicate or empty descriptor binding",
                ));
            }
            declarations.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(*count)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&declarations);
        // SAFETY: declarations remain live for this call and contain valid
        // storage-buffer descriptor types for the supplied SPIR-V contract.
        let descriptor_layout = match unsafe {
            self.inner
                .device
                .create_descriptor_set_layout(&layout_info, None)
        } {
            Ok(value) => value,
            Err(error) => return Err(HarError::vulkan("create descriptor set layout", error)),
        };
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_bytes);
        let mut pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_layout));
        if push_constant_bytes != 0 {
            pipeline_layout_info =
                pipeline_layout_info.push_constant_ranges(std::slice::from_ref(&push_range));
        }
        // SAFETY: descriptor_layout is live and the optional push range is live
        // through the call.
        let pipeline_layout = match unsafe {
            self.inner
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        } {
            Ok(value) => value,
            Err(error) => {
                unsafe {
                    self.inner
                        .device
                        .destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(HarError::vulkan("create pipeline layout", error));
            }
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: SPIR-V is a properly aligned u32 slice owned by the caller for
        // this call; Vulkan copies/validates the module during creation.
        let shader = match unsafe { self.inner.device.create_shader_module(&module_info, None) } {
            Ok(value) => value,
            Err(error) => {
                unsafe {
                    self.inner
                        .device
                        .destroy_pipeline_layout(pipeline_layout, None);
                    self.inner
                        .device
                        .destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(HarError::vulkan("create shader module", error));
            }
        };
        let entry_name = CString::new("main").expect("literal has no NUL");
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(required_subgroup.unwrap_or(32));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        if required_subgroup.is_some() {
            stage = stage.push_next(&mut subgroup);
        }
        if let Some(info) = spec_info {
            // SAFETY: spec_info and its backing spec_data live for this call.
            stage = stage.specialization_info(info);
        }
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)
            .base_pipeline_index(-1);
        // SAFETY: shader, pipeline_layout, and stage inputs are live and owned
        // by this device; the cache belongs to the same device when supplied.
        let result = unsafe {
            self.inner.device.create_compute_pipelines(
                cache.map_or(vk::PipelineCache::null(), |c| c.handle),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        };
        // SAFETY: shader module is no longer needed after pipeline creation.
        unsafe {
            self.inner.device.destroy_shader_module(shader, None);
        }
        let pipeline = match result {
            Ok(mut values) => values.remove(0),
            Err((_, error)) => {
                unsafe {
                    self.inner
                        .device
                        .destroy_pipeline_layout(pipeline_layout, None);
                    self.inner
                        .device
                        .destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(HarError::vulkan("create compute pipeline", error));
            }
        };
        Ok(Pipeline {
            device: self.inner.clone(),
            descriptor_layout,
            pipeline_layout,
            handle: pipeline,
            name: name.to_owned(),
            shader_hash: sha256_words(spirv),
            push_constant_bytes,
            bindings: declarations
                .iter()
                .map(|b| (b.binding, b.descriptor_count))
                .collect(),
        })
    }
    pub fn create_expert_slot_table(
        &self,
        slot_count: u32,
        slot_bytes: u64,
    ) -> HarResult<ExpertSlotTable> {
        if slot_count == 0 || slot_bytes == 0 || slot_bytes > usize::MAX as u64 / slot_count as u64
        {
            return Err(HarError::argument(
                "create expert slot table",
                "slot dimensions are zero or overflow host size",
            ));
        }
        let storage = self.create_buffer(
            slot_bytes as usize * slot_count as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryPreference::DeviceLocal,
            "har.expert.slot.storage",
        )?;
        let slots = (0..slot_count)
            .map(|slot_id| ExpertSlotRecord {
                snapshot: ExpertSlotSnapshot {
                    slot_id,
                    key: ExpertSlotKey {
                        layer: 0,
                        expert_id: 0,
                        projection: ExpertProjection::Gate,
                        generation: 0,
                    },
                    state: ExpertSlotState::Free,
                    upload_epoch: 0,
                    last_use_epoch: 0,
                    ready: false,
                },
                upload: None,
            })
            .collect();
        Ok(ExpertSlotTable {
            device: self.inner.clone(),
            storage,
            slot_bytes,
            slots: Mutex::new(slots),
            epoch: Mutex::new(1),
        })
    }
}

fn device_extension_present(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    wanted: &CStr,
) -> bool {
    // SAFETY: physical belongs to instance and the returned extension array is
    // owned by ash for the duration of this query.
    unsafe {
        instance
            .enumerate_device_extension_properties(physical)
            .map(|extensions| {
                extensions
                    .iter()
                    .any(|extension| CStr::from_ptr(extension.extension_name.as_ptr()) == wanted)
            })
            .unwrap_or(false)
    }
}

fn choose_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    preference: MemoryPreference,
) -> Option<u32> {
    let (required, preferred) = match preference {
        MemoryPreference::DeviceLocal => (
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        MemoryPreference::HostVisible => (
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ),
        MemoryPreference::HostVisibleDeviceLocal => (
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_COHERENT,
        ),
    };
    let mut fallback = None;
    for index in 0..properties.memory_type_count {
        if bits & (1 << index) == 0 {
            continue;
        }
        let flags = properties.memory_types[index as usize].property_flags;
        if flags.contains(required) {
            if flags.contains(preferred) {
                return Some(index);
            }
            fallback = Some(index);
        }
    }
    fallback
}

pub struct Allocation {
    device: Arc<DeviceInner>,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    properties: vk::MemoryPropertyFlags,
}
impl Allocation {
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn properties(&self) -> vk::MemoryPropertyFlags {
        self.properties
    }
    pub fn host_visible(&self) -> bool {
        self.properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> HarResult<()> {
        if !self.host_visible() {
            return Err(HarError::argument(
                "write allocation",
                "allocation is not host visible",
            ));
        }
        if offset > self.size as usize || bytes.len() > self.size as usize - offset {
            return Err(HarError::argument(
                "write allocation",
                "write range is outside allocation",
            ));
        }
        // SAFETY: range was checked against the allocation size; the mapped
        // pointer is valid until unmap_memory below, and bytes do not overlap it.
        let pointer = unsafe {
            vk_result(
                "map memory",
                self.device.device.map_memory(
                    self.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                ),
            )?
        } as *mut u8;
        // SAFETY: pointer + offset is within the mapped allocation and source
        // bytes are valid for bytes.len().
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.add(offset), bytes.len());
        }
        let flush_result = if !self
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            let atom = self.device.capabilities.non_coherent_atom_size.max(1);
            let start = (offset as u64 / atom) * atom;
            let end = (offset as u64 + bytes.len() as u64)
                .div_ceil(atom)
                .saturating_mul(atom)
                .min(self.size);
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(start)
                .size(end.saturating_sub(start));
            // SAFETY: range is aligned to the device atom and clipped to the
            // allocation; memory is currently mapped by this call.
            unsafe {
                vk_result(
                    "flush mapped memory",
                    self.device
                        .device
                        .flush_mapped_memory_ranges(std::slice::from_ref(&range)),
                )
            }
        } else {
            Ok(())
        };
        // SAFETY: this call closes the map opened above on the same device even
        // when the non-coherent flush reported a Vulkan error.
        unsafe {
            self.device.device.unmap_memory(self.memory);
        }
        flush_result
    }
    pub fn read(&self, offset: usize, size: usize) -> HarResult<Vec<u8>> {
        if !self.host_visible() {
            return Err(HarError::argument(
                "read allocation",
                "allocation is not host visible",
            ));
        }
        if offset > self.size as usize || size > self.size as usize - offset {
            return Err(HarError::argument(
                "read allocation",
                "read range is outside allocation",
            ));
        }
        // SAFETY: range was checked and the mapping is released before return.
        let pointer = unsafe {
            vk_result(
                "map memory",
                self.device.device.map_memory(
                    self.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                ),
            )?
        } as *const u8;
        let invalidate_result = if !self
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            let atom = self.device.capabilities.non_coherent_atom_size.max(1);
            let start = (offset as u64 / atom) * atom;
            let end = (offset as u64 + size as u64)
                .div_ceil(atom)
                .saturating_mul(atom)
                .min(self.size);
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(start)
                .size(end.saturating_sub(start));
            // SAFETY: range is aligned/clipped and memory is currently mapped.
            unsafe {
                vk_result(
                    "invalidate mapped memory",
                    self.device
                        .device
                        .invalidate_mapped_memory_ranges(std::slice::from_ref(&range)),
                )
            }
        } else {
            Ok(())
        };
        if let Err(error) = invalidate_result {
            // SAFETY: always close the mapping before propagating invalidation failure.
            unsafe {
                self.device.device.unmap_memory(self.memory);
            }
            return Err(error);
        }
        let mut output = vec![0u8; size];
        // SAFETY: pointer + offset points to at least size readable bytes.
        unsafe {
            ptr::copy_nonoverlapping(pointer.add(offset), output.as_mut_ptr(), size);
            self.device.device.unmap_memory(self.memory);
        }
        Ok(output)
    }
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: allocation is owned exactly once and child Buffer drops before
        // this field, so no live buffer refers to the memory.
        unsafe {
            self.device.device.free_memory(self.memory, None);
        }
    }
}

struct BufferInner {
    device: Arc<DeviceInner>,
    handle: vk::Buffer,
    allocation: Allocation,
    size: usize,
    usage: vk::BufferUsageFlags,
    name: String,
    generation: u64,
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: BufferInner owns this handle and its Allocation field is
        // declared after handle, so the buffer is destroyed first.
        unsafe {
            self.device.device.destroy_buffer(self.handle, None);
        }
    }
}
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}
impl Buffer {
    pub fn size(&self) -> usize {
        self.inner.size
    }
    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.inner.usage
    }
    pub fn generation(&self) -> u64 {
        self.inner.generation
    }
    pub fn name(&self) -> &str {
        &self.inner.name
    }
    pub fn allocation(&self) -> &Allocation {
        &self.inner.allocation
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> HarResult<()> {
        self.inner.allocation.write(offset, bytes)
    }
    pub fn read(&self, offset: usize, size: usize) -> HarResult<Vec<u8>> {
        self.inner.allocation.read(offset, size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProjection {
    Gate,
    Up,
    Down,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotKey {
    pub layer: u32,
    pub expert_id: u32,
    pub projection: ExpertProjection,
    pub generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSlotState {
    Free,
    Uploading,
    Ready,
    InFlight,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotSnapshot {
    pub slot_id: u32,
    pub key: ExpertSlotKey,
    pub state: ExpertSlotState,
    pub upload_epoch: u64,
    pub last_use_epoch: u64,
    pub ready: bool,
}
struct ExpertSlotRecord {
    snapshot: ExpertSlotSnapshot,
    upload: Option<TimelineEvent>,
}
pub struct ExpertSlotTable {
    device: Arc<DeviceInner>,
    storage: Buffer,
    slot_bytes: u64,
    slots: Mutex<Vec<ExpertSlotRecord>>,
    epoch: Mutex<u64>,
}
impl ExpertSlotTable {
    pub fn storage_buffer(&self) -> &Buffer {
        &self.storage
    }
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }
    pub fn slot_count(&self) -> u32 {
        self.slots.lock().map(|s| s.len() as u32).unwrap_or(0)
    }
    pub fn reserve(&self, key: ExpertSlotKey) -> HarResult<u32> {
        if key.generation == 0 {
            return Err(HarError::argument(
                "reserve expert slot",
                "generation zero is reserved",
            ));
        }
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| HarError::argument("reserve expert slot", "slot mutex poisoned"))?;
        if slots
            .iter()
            .any(|s| s.snapshot.state != ExpertSlotState::Free && s.snapshot.key == key)
        {
            return Err(HarError::argument(
                "reserve expert slot",
                "slot key is already resident or in flight",
            ));
        }
        let index = slots
            .iter()
            .position(|s| s.snapshot.state == ExpertSlotState::Free)
            .ok_or_else(|| {
                HarError::argument("reserve expert slot", "bounded slot table is full")
            })?;
        let epoch = self.next_epoch()?;
        slots[index].snapshot.key = key;
        slots[index].snapshot.state = ExpertSlotState::Uploading;
        slots[index].snapshot.upload_epoch = epoch;
        slots[index].snapshot.last_use_epoch = epoch;
        slots[index].snapshot.ready = false;
        slots[index].upload = None;
        Ok(index as u32)
    }
    pub fn mark_upload_submitted(&self, slot_id: u32, event: &TimelineEvent) -> HarResult<()> {
        self.check_event_device(event)?;
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| HarError::argument("mark expert upload", "slot mutex poisoned"))?;
        let slot = slots
            .get_mut(slot_id as usize)
            .ok_or_else(|| HarError::argument("mark expert upload", "slot ID is outside table"))?;
        if slot.snapshot.state != ExpertSlotState::Uploading {
            return Err(HarError::argument(
                "mark expert upload",
                "slot is not awaiting upload",
            ));
        }
        slot.upload = Some(event.clone());
        Ok(())
    }
    pub fn mark_ready(&self, slot_id: u32, event: &TimelineEvent) -> HarResult<()> {
        self.check_event_device(event)?;
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| HarError::argument("mark expert ready", "slot mutex poisoned"))?;
        let slot = slots
            .get_mut(slot_id as usize)
            .ok_or_else(|| HarError::argument("mark expert ready", "slot ID is outside table"))?;
        if slot.snapshot.state != ExpertSlotState::Uploading {
            return Err(HarError::argument(
                "mark expert ready",
                "slot is not uploading",
            ));
        }
        let upload = slot.upload.as_ref().ok_or_else(|| {
            HarError::argument("mark expert ready", "slot has no submitted upload event")
        })?;
        if upload.value != event.value {
            return Err(HarError::argument(
                "mark expert ready",
                "readiness event does not match upload event",
            ));
        }
        if !event.is_complete()? {
            return Err(HarError::argument(
                "mark expert ready",
                "upload event is not complete",
            ));
        }
        slot.snapshot.state = ExpertSlotState::Ready;
        slot.snapshot.ready = true;
        slot.upload = None;
        Ok(())
    }
    pub fn acquire_for_dispatch(&self, key: ExpertSlotKey) -> HarResult<ExpertSlotSnapshot> {
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| HarError::argument("acquire expert slot", "slot mutex poisoned"))?;
        let slot = slots
            .iter_mut()
            .find(|s| s.snapshot.key == key)
            .ok_or_else(|| {
                HarError::argument(
                    "acquire expert slot",
                    "projection-specific slot is not resident",
                )
            })?;
        if slot.snapshot.state != ExpertSlotState::Ready || !slot.snapshot.ready {
            return Err(HarError::argument(
                "acquire expert slot",
                "slot is not ready",
            ));
        }
        slot.snapshot.state = ExpertSlotState::InFlight;
        slot.snapshot.last_use_epoch = self.next_epoch()?;
        Ok(slot.snapshot)
    }
    pub fn release_from_dispatch(&self, slot_id: u32, event: &TimelineEvent) -> HarResult<()> {
        self.check_event_device(event)?;
        if !event.is_complete()? {
            return Err(HarError::argument(
                "release expert slot",
                "dispatch event is not complete",
            ));
        }
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| HarError::argument("release expert slot", "slot mutex poisoned"))?;
        let slot = slots
            .get_mut(slot_id as usize)
            .ok_or_else(|| HarError::argument("release expert slot", "slot ID is outside table"))?;
        if slot.snapshot.state != ExpertSlotState::InFlight {
            return Err(HarError::argument(
                "release expert slot",
                "slot is not in flight",
            ));
        }
        slot.snapshot.state = ExpertSlotState::Ready;
        slot.snapshot.ready = true;
        slot.snapshot.last_use_epoch = self.next_epoch()?;
        Ok(())
    }
    pub fn evict(&self, slot_id: u32, event: &TimelineEvent) -> HarResult<()> {
        self.check_event_device(event)?;
        if !event.is_complete()? {
            return Err(HarError::argument(
                "evict expert slot",
                "completion event is not complete",
            ));
        }
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| HarError::argument("evict expert slot", "slot mutex poisoned"))?;
        let slot = slots
            .get_mut(slot_id as usize)
            .ok_or_else(|| HarError::argument("evict expert slot", "slot ID is outside table"))?;
        if slot.snapshot.state == ExpertSlotState::InFlight {
            return Err(HarError::argument(
                "evict expert slot",
                "cannot evict an in-flight slot",
            ));
        }
        slot.snapshot.state = ExpertSlotState::Free;
        slot.snapshot.key = ExpertSlotKey {
            layer: 0,
            expert_id: 0,
            projection: ExpertProjection::Gate,
            generation: 0,
        };
        slot.snapshot.ready = false;
        slot.snapshot.last_use_epoch = self.next_epoch()?;
        slot.upload = None;
        Ok(())
    }
    pub fn snapshot(&self) -> Vec<ExpertSlotSnapshot> {
        self.slots
            .lock()
            .map(|s| s.iter().map(|r| r.snapshot).collect())
            .unwrap_or_default()
    }
    fn next_epoch(&self) -> HarResult<u64> {
        let mut epoch = self
            .epoch
            .lock()
            .map_err(|_| HarError::argument("expert slot epoch", "epoch mutex poisoned"))?;
        let value = *epoch;
        *epoch += 1;
        Ok(value)
    }
    fn check_event_device(&self, event: &TimelineEvent) -> HarResult<()> {
        if !Arc::ptr_eq(&self.device, &event.queue.device) {
            return Err(HarError::argument(
                "expert slot event",
                "event belongs to another device",
            ));
        }
        Ok(())
    }
}

pub struct PipelineCache {
    device: Arc<DeviceInner>,
    handle: vk::PipelineCache,
}
impl PipelineCache {
    pub fn data(&self) -> HarResult<Vec<u8>> {
        // SAFETY: cache is owned by this device; ash allocates the returned
        // byte vector after Vulkan reports its exact size.
        unsafe {
            vk_result(
                "get pipeline cache data",
                self.device.device.get_pipeline_cache_data(self.handle),
            )
        }
    }
    pub fn save(&self, path: impl AsRef<Path>) -> HarResult<()> {
        let data = self.data()?;
        fs::write(path, data).map_err(|e| HarError::load(format!("write pipeline cache: {e}")))
    }
}
impl Drop for PipelineCache {
    fn drop(&mut self) {
        // SAFETY: cache is owned exactly once by this wrapper.
        unsafe {
            self.device.device.destroy_pipeline_cache(self.handle, None);
        }
    }
}

pub struct Pipeline {
    device: Arc<DeviceInner>,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    handle: vk::Pipeline,
    name: String,
    shader_hash: String,
    push_constant_bytes: u32,
    bindings: Vec<(u32, u32)>,
}
impl Pipeline {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shader_hash(&self) -> &str {
        &self.shader_hash
    }
    pub fn push_constant_bytes(&self) -> u32 {
        self.push_constant_bytes
    }
}
impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: child pipeline objects are destroyed in reverse creation
        // order while DeviceInner is still kept alive by this struct.
        unsafe {
            self.device.device.destroy_pipeline(self.handle, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

struct QueueInner {
    device: Arc<DeviceInner>,
    kind: QueueKind,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    timeline: vk::Semaphore,
    next_value: Mutex<u64>,
}
impl Drop for QueueInner {
    fn drop(&mut self) {
        // SAFETY: queue is idle before child pools/semaphore are destroyed.
        unsafe {
            let _ = self.device.device.queue_wait_idle(self.queue);
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .device
                .destroy_command_pool(self.command_pool, None);
            self.device.device.destroy_semaphore(self.timeline, None);
        }
    }
}

pub struct Queue {
    inner: Arc<QueueInner>,
}
impl Queue {
    fn new(
        device: Arc<DeviceInner>,
        kind: QueueKind,
        family: u32,
        queue: vk::Queue,
    ) -> HarResult<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        // SAFETY: family is one of the selected queue families for device.
        let command_pool = vk_result("create command pool", unsafe {
            device.device.create_command_pool(&pool_info, None)
        })?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4096)];
        let descriptor_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(1024)
            .pool_sizes(&pool_sizes);
        // SAFETY: pool_sizes remains live through this call.
        let descriptor_pool =
            match unsafe { device.device.create_descriptor_pool(&descriptor_info, None) } {
                Ok(value) => value,
                Err(error) => {
                    unsafe {
                        device.device.destroy_command_pool(command_pool, None);
                    }
                    return Err(HarError::vulkan("create descriptor pool", error));
                }
            };
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        // SAFETY: type_info is a valid timeline semaphore pNext for this call.
        let timeline = match unsafe { device.device.create_semaphore(&semaphore_info, None) } {
            Ok(value) => value,
            Err(error) => {
                unsafe {
                    device.device.destroy_descriptor_pool(descriptor_pool, None);
                    device.device.destroy_command_pool(command_pool, None);
                }
                return Err(HarError::vulkan("create timeline semaphore", error));
            }
        };
        Ok(Self {
            inner: Arc::new(QueueInner {
                device,
                kind,
                queue,
                command_pool,
                descriptor_pool,
                timeline,
                next_value: Mutex::new(0),
            }),
        })
    }
    pub fn kind(&self) -> QueueKind {
        self.inner.kind
    }
    pub fn allocate_command_buffer(&self) -> HarResult<CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.inner.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: command pool is owned by this queue and allocation count is one.
        let mut values = vk_result("allocate command buffer", unsafe {
            self.inner.device.device.allocate_command_buffers(&info)
        })?;
        Ok(CommandBuffer {
            queue: self.inner.clone(),
            handle: values.remove(0),
            recording: false,
            ended: false,
        })
    }
    pub fn allocate_descriptor_set(&self, pipeline: &Pipeline) -> HarResult<DescriptorSet> {
        if !Arc::ptr_eq(&self.inner.device, &pipeline.device) {
            return Err(HarError::argument(
                "allocate descriptor set",
                "pipeline belongs to another device",
            ));
        }
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.inner.descriptor_pool)
            .set_layouts(std::slice::from_ref(&pipeline.descriptor_layout));
        // SAFETY: descriptor pool and layout are owned by this device.
        let mut values = vk_result("allocate descriptor set", unsafe {
            self.inner.device.device.allocate_descriptor_sets(&info)
        })?;
        Ok(DescriptorSet {
            queue: self.inner.clone(),
            pipeline: pipeline.handle,
            set: values.remove(0),
            declared_bindings: pipeline.bindings.iter().copied().collect(),
            buffers: BTreeMap::new(),
        })
    }
    pub fn submit_compute(
        &self,
        command: CommandBuffer,
        descriptor_sets: Vec<DescriptorSet>,
        waits: &[TimelineEvent],
        queries: Vec<TimestampQuery>,
    ) -> HarResult<ComputeTicket> {
        let event = self.submit(command, descriptor_sets, waits, queries)?;
        Ok(ComputeTicket { event })
    }
    pub fn submit_transfer(
        &self,
        command: CommandBuffer,
        waits: &[TimelineEvent],
        queries: Vec<TimestampQuery>,
    ) -> HarResult<TransferTicket> {
        let event = self.submit(command, Vec::new(), waits, queries)?;
        Ok(TransferTicket { event })
    }
    fn submit(
        &self,
        mut command: CommandBuffer,
        descriptor_sets: Vec<DescriptorSet>,
        waits: &[TimelineEvent],
        queries: Vec<TimestampQuery>,
    ) -> HarResult<TimelineEvent> {
        if !Arc::ptr_eq(&self.inner, &command.queue) {
            return Err(HarError::argument(
                "submit",
                "command buffer belongs to another queue",
            ));
        }
        if !command.ended {
            return Err(HarError::argument("submit", "command buffer was not ended"));
        }
        for wait in waits {
            if !Arc::ptr_eq(&self.inner.device, &wait.queue.device) {
                return Err(HarError::argument(
                    "submit",
                    "wait event belongs to another device",
                ));
            }
        }
        // Move raw command ownership into the submission keepalive.  The
        // caller's command object is consumed and cannot be reused stale.
        let raw_command = command.handle;
        command.handle = vk::CommandBuffer::null();
        let mut value_guard = self
            .inner
            .next_value
            .lock()
            .map_err(|_| HarError::argument("submit", "timeline counter mutex poisoned"))?;
        *value_guard += 1;
        let signal_value = *value_guard;
        drop(value_guard);
        let wait_semaphores = waits
            .iter()
            .map(|event| event.semaphore)
            .collect::<Vec<_>>();
        let wait_values = waits.iter().map(|event| event.value).collect::<Vec<_>>();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signal_values = [signal_value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&raw_command))
            .signal_semaphores(std::slice::from_ref(&self.inner.timeline))
            .push_next(&mut timeline);
        // SAFETY: all submit arrays and timeline values live for the call;
        // raw_command was allocated from this queue's pool and remains owned by
        // SubmittedResources until the event completes.
        if let Err(error) = unsafe {
            self.inner.device.device.queue_submit(
                self.inner.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )
        } {
            // SAFETY: submission did not take ownership on failure.
            unsafe {
                self.inner.device.device.free_command_buffers(
                    self.inner.command_pool,
                    std::slice::from_ref(&raw_command),
                );
            }
            return Err(HarError::vulkan("queue submit", error));
        }
        // Keep the command alive in an event-local wrapper; Queue::submit's
        // public ticket reconstructs this field immediately below.
        Ok(TimelineEvent {
            queue: self.inner.clone(),
            semaphore: self.inner.timeline,
            value: signal_value,
            keepalive: Arc::new(Mutex::new(Some(SubmittedRaw {
                queue: self.inner.clone(),
                command: raw_command,
                _descriptor_sets: descriptor_sets,
                _queries: queries,
            }))),
        })
    }
}

pub struct DescriptorSet {
    queue: Arc<QueueInner>,
    pipeline: vk::Pipeline,
    set: vk::DescriptorSet,
    declared_bindings: BTreeMap<u32, u32>,
    buffers: BTreeMap<u32, Arc<BufferInner>>,
}
impl DescriptorSet {
    #[allow(clippy::manual_is_multiple_of)]
    pub fn update_storage(
        &mut self,
        binding: u32,
        buffer: &Buffer,
        offset: usize,
        range: usize,
    ) -> HarResult<()> {
        if !self.declared_bindings.contains_key(&binding) {
            return Err(HarError::argument(
                "update descriptor",
                "binding is not declared by pipeline",
            ));
        }
        if !Arc::ptr_eq(&self.queue.device, &buffer.inner.device) {
            return Err(HarError::argument(
                "update descriptor",
                "buffer belongs to another device",
            ));
        }
        if !self
            .queue
            .device
            .capabilities
            .min_storage_buffer_offset_alignment
            .eq(&0)
            && (offset as u64
                % self
                    .queue
                    .device
                    .capabilities
                    .min_storage_buffer_offset_alignment)
                != 0
        {
            return Err(HarError::argument(
                "update descriptor",
                "storage buffer offset violates alignment",
            ));
        }
        if range == 0 || offset > buffer.size() || range > buffer.size() - offset {
            return Err(HarError::argument(
                "update descriptor",
                "descriptor range outside buffer",
            ));
        }
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.inner.handle)
            .offset(offset as u64)
            .range(range as u64);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));
        // SAFETY: descriptor set and buffer are owned by the same live device;
        // buffer_info remains live for this immediate update.
        unsafe {
            self.queue
                .device
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
        self.buffers.insert(binding, buffer.inner.clone());
        Ok(())
    }
    fn validate(&self) -> HarResult<()> {
        for buffer in self.buffers.values() {
            if buffer.generation != 1 {
                return Err(HarError::argument(
                    "submit descriptor",
                    "stale buffer generation",
                ));
            }
        }
        Ok(())
    }
}
impl Drop for DescriptorSet {
    fn drop(&mut self) {
        // SAFETY: set is allocated from queue.descriptor_pool and no ticket can
        // hold this wrapper after Drop; tickets retain DescriptorSet explicitly.
        unsafe {
            let _ = self
                .queue
                .device
                .device
                .free_descriptor_sets(self.queue.descriptor_pool, std::slice::from_ref(&self.set));
        }
    }
}

pub struct CommandBuffer {
    queue: Arc<QueueInner>,
    handle: vk::CommandBuffer,
    recording: bool,
    ended: bool,
}
impl CommandBuffer {
    pub fn begin(&mut self) -> HarResult<()> {
        if self.recording || self.ended {
            return Err(HarError::argument(
                "begin command buffer",
                "command buffer is not reusable",
            ));
        }
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: handle was allocated from queue.command_pool and is not
        // concurrently recorded by another HAR wrapper.
        vk_result("begin command buffer", unsafe {
            self.queue
                .device
                .device
                .begin_command_buffer(self.handle, &info)
        })?;
        self.recording = true;
        Ok(())
    }
    pub fn bind_pipeline(&mut self, pipeline: &Pipeline) -> HarResult<()> {
        if !self.recording || !Arc::ptr_eq(&self.queue.device, &pipeline.device) {
            return Err(HarError::argument(
                "bind pipeline",
                "command buffer or pipeline is invalid",
            ));
        }
        // SAFETY: pipeline is created on the same device and command buffer is recording.
        unsafe {
            self.queue.device.device.cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle,
            );
        }
        Ok(())
    }
    pub fn bind_descriptor_set(
        &mut self,
        set: &DescriptorSet,
        pipeline: &Pipeline,
    ) -> HarResult<()> {
        if !self.recording
            || !Arc::ptr_eq(&self.queue.device, &set.queue.device)
            || !Arc::ptr_eq(&self.queue.device, &pipeline.device)
            || set.pipeline != pipeline.handle
        {
            return Err(HarError::argument(
                "bind descriptor set",
                "descriptor/pipeline/command device mismatch",
            ));
        }
        set.validate()?;
        // SAFETY: set was allocated for pipeline.pipeline_layout and both are
        // live on the command buffer's device.
        unsafe {
            self.queue.device.device.cmd_bind_descriptor_sets(
                self.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                std::slice::from_ref(&set.set),
                &[],
            );
        }
        Ok(())
    }
    pub fn push_constants(&mut self, pipeline: &Pipeline, bytes: &[u8]) -> HarResult<()> {
        if bytes.len() != pipeline.push_constant_bytes as usize {
            return Err(HarError::argument(
                "push constants",
                "byte count does not match pipeline",
            ));
        }
        if !self.recording {
            return Err(HarError::argument(
                "push constants",
                "command buffer is not recording",
            ));
        }
        // SAFETY: bytes remains live for the immediate recording call and its
        // size is bounded by pipeline creation validation.
        unsafe {
            self.queue.device.device.cmd_push_constants(
                self.handle,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
        Ok(())
    }
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> HarResult<()> {
        if !self.recording || x == 0 || y == 0 || z == 0 {
            return Err(HarError::argument(
                "dispatch",
                "invalid dimensions or recording state",
            ));
        }
        // SAFETY: command buffer is recording and dimensions are non-zero.
        unsafe {
            self.queue.device.device.cmd_dispatch(self.handle, x, y, z);
        }
        Ok(())
    }
    pub fn copy_buffer(
        &mut self,
        source: &Buffer,
        destination: &Buffer,
        source_offset: usize,
        destination_offset: usize,
        bytes: usize,
    ) -> HarResult<()> {
        if !self.recording
            || !Arc::ptr_eq(&self.queue.device, &source.inner.device)
            || !Arc::ptr_eq(&self.queue.device, &destination.inner.device)
        {
            return Err(HarError::argument(
                "copy buffer",
                "invalid recording state or device",
            ));
        }
        if !source.usage().contains(vk::BufferUsageFlags::TRANSFER_SRC)
            || !destination
                .usage()
                .contains(vk::BufferUsageFlags::TRANSFER_DST)
        {
            return Err(HarError::argument(
                "copy buffer",
                "transfer usage was not declared",
            ));
        }
        if bytes == 0
            || source_offset > source.size()
            || destination_offset > destination.size()
            || bytes > source.size() - source_offset
            || bytes > destination.size() - destination_offset
        {
            return Err(HarError::argument(
                "copy buffer",
                "copy range outside buffer",
            ));
        }
        let region = vk::BufferCopy::default()
            .src_offset(source_offset as u64)
            .dst_offset(destination_offset as u64)
            .size(bytes as u64);
        // SAFETY: source/destination handles and region are valid for this
        // device and command buffer recording scope.
        unsafe {
            self.queue.device.device.cmd_copy_buffer(
                self.handle,
                source.inner.handle,
                destination.inner.handle,
                std::slice::from_ref(&region),
            );
        }
        Ok(())
    }
    pub fn reset_query(&mut self, query: &TimestampQuery) -> HarResult<()> {
        if !self.recording || !Arc::ptr_eq(&self.queue.device, &query.inner.device) {
            return Err(HarError::argument(
                "reset query",
                "invalid query or recording state",
            ));
        }
        // SAFETY: query pool belongs to the same device and the command buffer is recording.
        unsafe {
            self.queue.device.device.cmd_reset_query_pool(
                self.handle,
                query.inner.pool,
                0,
                query.inner.count,
            );
        }
        Ok(())
    }
    pub fn write_timestamp(
        &mut self,
        query: &TimestampQuery,
        index: u32,
        stage: vk::PipelineStageFlags,
    ) -> HarResult<()> {
        if !self.recording
            || index >= query.inner.count
            || !Arc::ptr_eq(&self.queue.device, &query.inner.device)
        {
            return Err(HarError::argument(
                "write timestamp",
                "invalid query or index",
            ));
        }
        // SAFETY: query pool belongs to the same device and index is bounded.
        unsafe {
            self.queue.device.device.cmd_write_timestamp(
                self.handle,
                stage,
                query.inner.pool,
                index,
            );
        }
        Ok(())
    }
    pub fn end(&mut self) -> HarResult<()> {
        if !self.recording || self.ended {
            return Err(HarError::argument(
                "end command buffer",
                "command buffer is not recording",
            ));
        }
        // SAFETY: all commands were recorded between begin/end on this handle.
        vk_result("end command buffer", unsafe {
            self.queue.device.device.end_command_buffer(self.handle)
        })?;
        self.recording = false;
        self.ended = true;
        Ok(())
    }
}
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if self.handle != vk::CommandBuffer::null() {
            // SAFETY: handle came from this queue's command pool and is freed
            // only after the owning submission keepalive is dropped.
            unsafe {
                self.queue.device.device.free_command_buffers(
                    self.queue.command_pool,
                    std::slice::from_ref(&self.handle),
                );
            }
        }
    }
}

struct SubmittedRaw {
    queue: Arc<QueueInner>,
    command: vk::CommandBuffer,
    _descriptor_sets: Vec<DescriptorSet>,
    _queries: Vec<TimestampQuery>,
}
impl Drop for SubmittedRaw {
    fn drop(&mut self) {
        // SAFETY: the owning TimelineEvent waits before releasing this record;
        // therefore command execution is complete before pool free.
        unsafe {
            self.queue
                .device
                .device
                .free_command_buffers(self.queue.command_pool, std::slice::from_ref(&self.command));
        }
    }
}
pub struct TimelineEvent {
    queue: Arc<QueueInner>,
    semaphore: vk::Semaphore,
    value: u64,
    keepalive: Arc<Mutex<Option<SubmittedRaw>>>,
}
impl Clone for TimelineEvent {
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
            semaphore: self.semaphore,
            value: self.value,
            keepalive: self.keepalive.clone(),
        }
    }
}
impl TimelineEvent {
    pub fn value(&self) -> u64 {
        self.value
    }
    pub fn is_complete(&self) -> HarResult<bool> {
        // SAFETY: semaphore belongs to the live queue device.
        let value = vk_result("get timeline value", unsafe {
            self.queue
                .device
                .device
                .get_semaphore_counter_value(self.semaphore)
        })?;
        Ok(value >= self.value)
    }
    pub fn wait(&self) -> HarResult<()> {
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.semaphore))
            .values(std::slice::from_ref(&self.value));
        // SAFETY: wait references a live timeline semaphore and value.
        vk_result("wait timeline event", unsafe {
            self.queue.device.device.wait_semaphores(&wait, u64::MAX)
        })
    }
}
impl Drop for TimelineEvent {
    fn drop(&mut self) {
        let _ = self.wait();
        // Releasing the last Arc runs SubmittedRaw::drop only after the wait.
        let _ = self.keepalive.lock().map(|mut slot| slot.take());
    }
}

pub struct ComputeTicket {
    event: TimelineEvent,
}
impl ComputeTicket {
    pub fn event(&self) -> TimelineEvent {
        self.event.clone()
    }
    pub fn wait(&self) -> HarResult<()> {
        self.event.wait()
    }
}
impl Drop for ComputeTicket {
    fn drop(&mut self) {
        let _ = self.event.wait();
    }
}

pub struct TransferTicket {
    event: TimelineEvent,
}
impl TransferTicket {
    pub fn event(&self) -> TimelineEvent {
        self.event.clone()
    }
    pub fn wait(&self) -> HarResult<()> {
        self.event.wait()
    }
}
impl Drop for TransferTicket {
    fn drop(&mut self) {
        let _ = self.event.wait();
    }
}

#[derive(Clone)]
pub struct TimestampQuery {
    inner: Arc<TimestampQueryInner>,
}
struct TimestampQueryInner {
    device: Arc<DeviceInner>,
    pool: vk::QueryPool,
    count: u32,
}
impl TimestampQuery {
    pub fn count(&self) -> u32 {
        self.inner.count
    }
    pub fn raw_ticks(&self) -> HarResult<Vec<u64>> {
        let mut values = vec![0u64; self.inner.count as usize];
        // SAFETY: values has exactly count 64-bit entries and WAIT makes the
        // result valid before returning.
        vk_result("get timestamp query", unsafe {
            self.inner.device.device.get_query_pool_results(
                self.inner.pool,
                0,
                &mut values,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        })?;
        Ok(values)
    }
    pub fn elapsed_ns(&self) -> HarResult<u64> {
        let values = self.raw_ticks()?;
        if values.len() < 2 {
            return Err(HarError::argument(
                "elapsed timestamp",
                "at least two queries are required",
            ));
        }
        Ok(((values[1].wrapping_sub(values[0]) as f64)
            * self.inner.device.capabilities.timestamp_period_ns as f64) as u64)
    }
}
impl Drop for TimestampQueryInner {
    fn drop(&mut self) {
        // SAFETY: pool is owned by the final Arc and no pending command may use
        // it because submitted tickets retain a TimestampQuery clone.
        unsafe {
            self.device.device.destroy_query_pool(self.pool, None);
        }
    }
}

fn sha256_words(words: &[u32]) -> String {
    // A compact SHA-256 implementation would duplicate workspace logic. Use
    // the standard library-free FNV-derived label only internally?  No: shader
    // identity must be cryptographic, so call the small audited implementation
    // below, operating on the exact little-endian SPIR-V bytes.
    let mut hasher = Sha256::new();
    for word in words {
        hasher.update(&word.to_le_bytes());
    }
    hasher.finish()
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    used: usize,
    length: u64,
}
impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            used: 0,
            length: 0,
        }
    }
    fn update(&mut self, data: &[u8]) {
        self.length += data.len() as u64;
        let mut input = data;
        while !input.is_empty() {
            let take = (64 - self.used).min(input.len());
            self.buffer[self.used..self.used + take].copy_from_slice(&input[..take]);
            self.used += take;
            input = &input[take..];
            if self.used == 64 {
                self.block();
                self.used = 0;
            }
        }
    }
    fn block(&mut self) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            *word = u32::from_be_bytes([
                self.buffer[4 * i],
                self.buffer[4 * i + 1],
                self.buffer[4 * i + 2],
                self.buffer[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let a = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let b = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(a)
                .wrapping_add(w[i - 7])
                .wrapping_add(b);
        }
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
    fn finish(mut self) -> String {
        let bits = self.length * 8;
        self.buffer[self.used] = 0x80;
        self.used += 1;
        if self.used > 56 {
            for x in &mut self.buffer[self.used..] {
                *x = 0;
            }
            self.block();
            self.used = 0;
        }
        for x in &mut self.buffer[self.used..56] {
            *x = 0;
        }
        self.buffer[56..64].copy_from_slice(&bits.to_be_bytes());
        self.block();
        self.state.iter().map(|x| format!("{x:08x}")).collect()
    }
}

// The following raw handles are intentionally not public.  Callers can only
// submit through the lifetime-checked wrappers above.
#[allow(dead_code)]
fn _raw_handle_is_private(_: vk::Buffer) {}

#[cfg(test)]
mod tests {
    use super::sha256_words;

    #[test]
    fn shader_hash_uses_little_endian_spirv_bytes() {
        assert_eq!(
            sha256_words(&[0x0723_0203]),
            "b263db113857145cd1ec390021310f185f50bd5e3027f4111c1c5d9b7c0890a4"
        );
    }
}

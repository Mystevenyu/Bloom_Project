#pragma once
#pragma once
#include"Header.h"

#include <tuple>

struct LoadMeshInfo
{
	uint32_t materialIndex;
	lut::Buffer loadIndices;
	lut::Buffer loadData;

	std::size_t loadIndexCount;
};

struct NewMaterialInfo
{
	glm::vec3 baseColor;
	float roughness;
	glm::vec3 emissiveColor;
	float metalness;

	int size = 3;
};
static_assert(sizeof(NewMaterialInfo) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
static_assert(sizeof(NewMaterialInfo) % 4 == 0, "SceneUniform size must be multiple of 4 bytes");

NewMaterialInfo from_baked_material(const BakedMaterialInfo& bakedMaterial);
struct LoadModel
{
	std::vector<LoadMeshInfo>meshInfo;
	std::vector<VkDescriptorSet>descriptorInfo;
	std::vector<lut::Image>images;
	std::vector<lut::ImageView> imageViews;
	std::vector<NewMaterialInfo>newMaterialInfo;
};

LoadModel create_loaded_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, VkDescriptorPool& dpool, VkCommandPool& aCmdPool, VkDescriptorSetLayout& aObjectLayout, VkSampler& aSampler, BakedModel const& ponzaModel);
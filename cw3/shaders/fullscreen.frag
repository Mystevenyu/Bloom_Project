#version 450
#extension GL_KHR_vulkan_glsl: enable

layout (location = 0) in vec2 inUV;

layout (set = 0, binding = 0) uniform sampler2D brightTexture;

layout (set = 1, binding = 0) uniform sampler2D normalTexture;

layout (location = 0) out vec4 oColor;






void main()
{
	vec3 pixelBrightColor = texture(brightTexture, inUV).rgb;

	vec3 pixelNormalColor = texture(normalTexture, inUV).rgb;

	oColor = vec4(pixelNormalColor + pixelBrightColor, 1.0f);
}
#version 450
#extension GL_KHR_vulkan_glsl: enable

layout (location = 0) in vec2 inUV;

layout (set = 0, binding = 0) uniform sampler2D uTexColor;

layout (location = 0) out vec4 oColor;

float M_PI = 3.1415926535;

const int GAUSSIAN_SAMPLES = 11;
const float SIGMA = 9.0;

float gaussian(float x, float sigma) {
    return (1.0 / (sqrt(2.0 * M_PI) * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

//To perform more precise edge handling on texture coordinates, it first restricts the coordinates to the range [0, 1], 
//and then moves the coordinates half a pixel towards the center to avoid undesired edge effects when sampling edge pixels.
vec4 textureClampToEdge(sampler2D tex, vec2 uv) {
    vec2 clampedUV = clamp(uv, 0.0, 1.0);
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 halfPixel = vec2(0.5) / texSize;
    clampedUV = mix(halfPixel, vec2(1.0) - halfPixel, clampedUV);
    return texture(tex, clampedUV);
}

void main() {
    vec3 pixelBrightColor = texture(uTexColor, inUV).rgb;

    vec2 tex_offset = 1.0 / textureSize(uTexColor, 0);
    vec3 result = pixelBrightColor * gaussian(0.0, SIGMA); // Current fragment

    for (int i = 1; i <= GAUSSIAN_SAMPLES; i+=2) {
        float weight1 = gaussian(float(i),SIGMA);
        float weight2 = gaussian(float(i + 1),SIGMA);
        float combinedWeight = weight1 + weight2;

        vec3 sample1 = textureClampToEdge(uTexColor, inUV + vec2(tex_offset.x * i, 0.0)).rgb;
        vec3 sample2 = textureClampToEdge(uTexColor, inUV + vec2(tex_offset.x * (i + 1), 0.0)).rgb;
        vec3 combinedSample = mix(sample1, sample2, weight2 / combinedWeight);

        result += combinedSample * combinedWeight;

        sample1 = textureClampToEdge(uTexColor, inUV - vec2(tex_offset.x * i, 0.0)).rgb;
        sample2 = textureClampToEdge(uTexColor, inUV - vec2(tex_offset.x * (i + 1), 0.0)).rgb;
        combinedSample = mix(sample1, sample2, weight2 / combinedWeight);

        result += combinedSample * combinedWeight;
    }

    oColor = vec4(result, 1.0);
}
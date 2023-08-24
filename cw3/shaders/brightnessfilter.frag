#version 450
#extension GL_KHR_vulkan_glsl: enable

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 oNormal;
layout (location = 2) in vec3 oPosition;
layout (location = 3) in vec3 oCameraPos;
layout (location = 4) in vec3 oLightPos[3];
layout (location = 7) in vec3 oLightColor[3];

layout(set = 1,binding = 0) uniform sampler2D baseColorTex;//baseColor
layout(set = 1,binding = 1) uniform sampler2D roughnessTex;
layout(set = 1,binding = 2) uniform sampler2D metalnessTex;

layout(set = 2, binding = 0) uniform UMaterial {
    vec3 uBaseColor;
	float uRoughness;
    vec3 uEmissiveColor;
    float uMetalness;
} material;


layout (location = 0) out vec4 outColor;

float M_PI = 3.1415926;



//light


float BlinnPhongDist(float shininess,vec3 surfaceNor,vec3 halfVec)
{
	float surNDotHalfV = max(dot(surfaceNor,halfVec),0.0);
	return ((shininess + 2) / (2 * M_PI)) * pow(surNDotHalfV,shininess);
}

float CookTorranceModel(vec3 surfaceNor,vec3 halfVec,vec3 lightDir,vec3 viewDir)
{
	float nhnv = 2 * (max(dot(surfaceNor,halfVec),0.f) * max(dot(surfaceNor,viewDir),0.f)) / (dot(viewDir,halfVec) + 0.0001);
	float nhnl = 2 * (max(dot(surfaceNor,halfVec),0.f) * max(dot(surfaceNor,lightDir),0.f)) / (dot(viewDir,halfVec) + 0.0001);
	return min(1.0,min(nhnv,nhnl));
}



void main()
{
	vec4 ambientColor = vec4(0.02f, 0.02f, 0.02f, 1.0f);
	vec3 result = vec3(0, 0, 0);
	for(int i = 0;i < 3; i++)
	{
	vec3 lightDir = normalize(oLightPos[i] - oPosition);

    //view direction, half vector
    vec3 V = normalize(oCameraPos - oPosition);
    vec3 H = normalize(V + lightDir);
    vec3 N = normalize(oNormal);

    float shininess = (2.f / (pow(material.uRoughness, 4) + 0.0001)) - 2.f;

    vec3 F0 = ((1 - material.uMetalness) * vec3(0.04f)) + material.uMetalness * material.uBaseColor; 

    //Ldiffuse¡¢G¡¢D¡¢F
    vec3 F = F0 + (1.f - F0) * pow((1.f - dot(H,V)), 5);    //Fresnel schlick approximation
    float D = BlinnPhongDist(shininess,N,H);
    float G = CookTorranceModel(N,H,lightDir,V);
    vec3 Ldiffuse =  (material.uBaseColor / M_PI) * (vec3(1.f) - F) * (1.f - material.uMetalness);
	//BRDF
	vec3 up = D * F * G;
	float down = 4.0 * max(dot(N,V),0) * max(dot(N,lightDir),0) + 0.0001;
	vec3 BRDF = Ldiffuse + up / down;


	float Clight = max(dot(N,lightDir),0);
	vec3 L0 = (BRDF * oLightColor[i]) * Clight;

	result += L0;
	}

	vec3 ambient = vec3(ambientColor) * material.uBaseColor;

	vec3 emissive = vec3(material.uEmissiveColor) * material.uBaseColor;

	vec4 pixelColor = vec4(emissive + ambient + result,1);

	float flag = pixelColor.x + pixelColor.y + pixelColor.z;

	if(flag >= 0.5f)
	{
		outColor = pixelColor;
	}
	else
	{
		outColor = vec4(0.f,0.f,0.f,1.f);
	}

	

}


#include "volumeTexture.cuh"
#include "cudaUtils.cuh"


void VolumeTexture::loadVolume(const std::vector<uint8_t>& hostData, int width, int height, int depth) {
	// preparing the cuda array to get the data
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
	cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	CHECK_CUDA(cudaMalloc3DArray(&contentArray, &channelDesc, volumeSize));
	//*****************************************

	// copying the data from the linear host array
	cudaMemcpy3DParms copyParms = { 0 };
	copyParms.srcPtr = make_cudaPitchedPtr((void*)(hostData.data()), width * sizeof(uint8_t), width, height);
	copyParms.dstArray = contentArray;
	copyParms.extent = volumeSize;
	copyParms.kind = cudaMemcpyHostToDevice;

	CHECK_CUDA(cudaMemcpy3D(&copyParms));
	//*******************************************

	// create the texture object
	
	// 1. prepare rsource description
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = contentArray;
	//********************************

	// 2. texture description
	struct cudaTextureDesc textureDesc;
	memset(&textureDesc, 0, sizeof(textureDesc));

	textureDesc.addressMode[0] = cudaAddressModeClamp;
	textureDesc.addressMode[1] = cudaAddressModeClamp;
	textureDesc.addressMode[2] = cudaAddressModeClamp;

	textureDesc.filterMode = cudaFilterModeLinear;

	textureDesc.readMode = cudaReadModeNormalizedFloat;
	textureDesc.normalizedCoords = true;

	CHECK_CUDA(cudaCreateTextureObject(&textureObject, &resDesc, &textureDesc, NULL));
	//*************************

	//***************************
}
void VolumeTexture::cleanup() {
	// ensure that device finshed its job
	cudaDeviceSynchronize();
	//********************

	// destroy the texture;
	if (textureObject != 0) {
		cudaDestroyTextureObject(textureObject);
		textureObject = 0;
	}
	//********************

	// free the cuda array
	if (contentArray != nullptr) {
		cudaFreeArray(contentArray);
		contentArray = nullptr;
	}
	//********************
}

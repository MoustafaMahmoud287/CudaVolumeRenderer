#ifndef VOLUME_LOADER_H
#define VOLUME_LOADER_H

#include <fstream>
#include <vector>
#include <iostream>
#include <string>

class VolumeLoader {
public:
    std::vector<uint8_t> data; // to read the 8bit data from the raw file
    int width, height, depth; // to store the dimenations of the loaded volume

    VolumeLoader(int w, int h, int d) : width(w), height(h), depth(d) {} // save the volume dims

    // take the pass and load the volume if everything is right
    bool load(const std::string& filepath) {
        //start from the end to get the size
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);

        // check if the file is opened
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return false;
        }

        // get the file size
        std::streamsize fileSize = file.tellg();

        // get the expected size from the dims
        size_t expectedBytes = width * height * depth * sizeof(uint8_t);

        // make sure that the expected and actual size is the same
        if (static_cast<size_t>(fileSize) != expectedBytes) {
            std::cerr << "Error: File size mismatch! Expected " << expectedBytes
                << " bytes, but got " << fileSize << " bytes." << std::endl;
            return false;
        }

        // move the file pointer to the begining to start reading
        file.seekg(0, std::ios::beg);

        // resize the data vector to the desierd size;
        data.resize(expectedBytes);

        // start the reading process
        if (!file.read(reinterpret_cast<char*>(data.data()), expectedBytes)) {
            std::cerr << "Error: Failed to read data stream." << std::endl;
            return false;
        }

        // used for console check
        std::cout << "[Success] Loaded volume. Size: " << fileSize << " bytes." << std::endl;
        return true;
    }
};

#endif

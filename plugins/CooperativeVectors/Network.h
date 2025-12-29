#pragma once

#include <vector>
#include <nvrhi/utils.h>

#include "Fluxel.h"
#include <donut/core/vfs/VFS.h>

NAMESPACE_BEGIN(fluxel)

enum class MatrixLayout
{
    RowMajor,
    ColumnMajor,
    InferencingOptimal,
    TrainingOptimal,
};

enum class Precision
{
    F16,
    F32
};

struct NetworkArchitecture
{
    uint32_t numHiddenLayers = 0;
    uint32_t inputNeurons = 0;
    uint32_t hiddenNeurons = 0;
    uint32_t outputNeurons = 0;
    Precision weightPrecision = Precision::F16;
    Precision biasPrecision = Precision::F16;
};

struct NetworkLayer
{
    uint32_t inputs = 0; ///< Columns in the weight matrix.
    uint32_t outputs = 0; ///< Rows in the weight matrix.
    size_t weightSize = 0; ///< Size of the weight matrix in bytes.
    size_t biasSize = 0; ///< Size of the bias vector in bytes.
    uint32_t weightOffset = 0; ///< Offset to the weights in bytes.
    uint32_t biasOffset = 0; ///< Offset to the biases in bytes.
};

struct NetworkLayout
{
    MatrixLayout matrixLayout = MatrixLayout::RowMajor;
    Precision matrixPrecision = Precision::F16;
    size_t networkSize = 0;
    std::vector<NetworkLayer> networkLayers;
};

constexpr size_t GetSize(Precision precision)
{
    switch (precision)
    {
    case Precision::F16:
        return sizeof(uint16_t); // 2 bytes
    case Precision::F32:
        return sizeof(float);
    default:
        return 0; // Should not get here
    }
}

class NetworkUtilities
{
public:
    NetworkUtilities(nvrhi::DeviceHandle device);
    ~NetworkUtilities()
    {
    }

    bool ValidateNetworkArchitecture(NetworkArchitecture const& netArch);

    // Create host side network layout.
    NetworkLayout CreateHostNetworkLayout(NetworkArchitecture const& netArch);

    // Set the weights and bias size / offsets for each layer in the network.
    void SetNetworkLayerSizes(NetworkLayout& layout);

    // Returns a updated network layout where the weights and bias size / offsets have been update
    // for the new matrix layout
    // Can be device optimal matrix layout
    NetworkLayout GetNewMatrixLayout(NetworkLayout const& srcLayout, MatrixLayout newMatrixLayout);

    // Converts weights and bias buffers from src layout to the dst layout.
    // Both buffers must be device side.
    // Both networks must be of the same network layout, only differing in MatrixLayout
    void ConvertWeights(NetworkLayout const& srcLayout,
                        NetworkLayout const& dstLayout,
                        nvrhi::BufferHandle srcBuffer,
                        uint64_t srcBufferOffset,
                        nvrhi::BufferHandle dstBuffer,
                        uint64_t dstBufferOffset,
                        nvrhi::DeviceHandle device,
                        nvrhi::CommandListHandle commandList);

private:
    nvrhi::DeviceHandle m_device;
};

// Represent a host side neural network.
// Stores the network layout and parameters.
// Functionality to initialize a network to starting values or load from file.
// Also write parameters back to file
class HostNetwork
{
public:
    HostNetwork(std::shared_ptr<NetworkUtilities> networkUtils);
    ~HostNetwork(){};

    // Create host side network from provided architecture with initial values.
    bool Initialise(const NetworkArchitecture& netArch);

    // Create host side network of provided architecture and initial values from a json file.
    bool InitialiseFromJson(donut::vfs::IFileSystem& fs, const std::string& fileName);
    // Create host side network of provided architecture and initial values from a file.
    bool InitialiseFromFile(const std::string& fileName);
    // Create host side network from an existing network.
    bool InitialiseFromNetwork(HostNetwork const& network);
    // Write the current network and parameters to file.
    bool WriteToFile(const std::string& fileName);
    // Convert device layout to host layout and update the host side parameters.
    void UpdateFromBufferToFile(nvrhi::BufferHandle hostLayoutBuffer,
                                nvrhi::BufferHandle deviceLayoutBuffer,
                                NetworkLayout const& hostLayout,
                                NetworkLayout const& deviceLayout,
                                const std::string& fileName,
                                nvrhi::DeviceHandle device,
                                nvrhi::CommandListHandle commandList);

    const NetworkArchitecture& GetNetworkArchitecture() const
    {
        return m_networkArchitecture;
    }

    const std::vector<uint8_t>& GetNetworkParams() const
    {
        return m_networkParams;
    }

    const NetworkLayout& GetNetworkLayout() const
    {
        return m_networkLayout;
    }

private:
    std::shared_ptr<NetworkUtilities> m_networkUtils;
    NetworkArchitecture m_networkArchitecture;
    std::vector<uint8_t> m_networkParams;
    NetworkLayout m_networkLayout;
};

NAMESPACE_END(fluxel)

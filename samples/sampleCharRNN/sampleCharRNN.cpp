/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

//!
//! sampleCharRNN.cpp
//! This file contains the implementation of the char_rnn sample.
//! It uses weights from a trained TensorFlow model and creates the network
//! using the TensorRT network definition API
//! It can be run with the following command line:
//! Command: ./sample_char_rnn [-h or --help] [-d or --datadir=<path to data directory>]
//!

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

#include "NvInfer.h"
#include "NvUtils.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include "logger.h"

const std::string gSampleName = "TensorRT.sample_char_rnn";

// The model used by this sample was trained using github repository:
// https://github.com/crazydonkey200/tensorflow-char-rnn
//
// The data set used: tensorflow-char-rnn/data/tiny_shakespeare.txt
//
// The command used to train:
// python train.py --data_file=data/tiny_shakespeare.txt --num_epochs=100 --num_layer=2 --hidden_size=512 --embedding_size=512 --dropout=.5
//
// Epochs trained: 100
// Test perplexity: 4.940
//
// Layer0 and Layer1 weights matrices are added as RNNW_L0_NAME and RNNW_L1_NAME, respectively.
// Layer0 and Layer1 bias are added as RNNB_L0_NAME and RNNB_L1_NAME, respectively.
// Embedded is added as EMBED_NAME.
// fc_w is added as FCW_NAME.
// fc_b is added as FCB_NAME.
struct SampleCharRNNWeightNames
{
    const std::string RNNW_L0_NAME{"rnn_multi_rnn_cell_cell_0_basic_lstm_cell_kernel"};
    const std::string RNNB_L0_NAME{"rnn_multi_rnn_cell_cell_0_basic_lstm_cell_bias"};
    const std::string RNNW_L1_NAME{"rnn_multi_rnn_cell_cell_1_basic_lstm_cell_kernel"};
    const std::string RNNB_L1_NAME{"rnn_multi_rnn_cell_cell_1_basic_lstm_cell_bias"};
    const std::string FCW_NAME{"softmax_softmax_w"};
    const std::string FCB_NAME{"softmax_softmax_b"};
    const std::string EMBED_NAME{"embedding"};

    std::unordered_set<std::string> names = {{RNNW_L0_NAME, RNNB_L0_NAME, RNNW_L1_NAME,
                                              RNNB_L1_NAME, FCW_NAME, FCB_NAME, EMBED_NAME}};
};

struct SampleCharRNNBindingNames
{
    const char* INPUT_BLOB_NAME{"data"};
    const char* HIDDEN_IN_BLOB_NAME{"hiddenIn"};
    const char* CELL_IN_BLOB_NAME{"cellIn"};
    const char* HIDDEN_OUT_BLOB_NAME{"hiddenOut"};
    const char* CELL_OUT_BLOB_NAME{"cellOut"};
    const char* OUTPUT_BLOB_NAME{"pred"};
    const char* SEQ_LEN_IN_BLOB_NAME{"seqLen"};
};

struct SampleCharRNNMaps
{
    // A mapping from character to index used by the tensorflow model.
    const std::map<char, int> charToID{{'\n', 0}, {'!', 1}, {' ', 2}, {'$', 3}, {'\'', 4}, {'&', 5}, {'-', 6}, {',', 7}, {'.', 8}, {'3', 9}, {';', 10}, {':', 11}, {'?', 12}, {'A', 13}, {'C', 14}, {'B', 15}, {'E', 16}, {'D', 17}, {'G', 18}, {'F', 19}, {'I', 20}, {'H', 21}, {'K', 22}, {'J', 23}, {'M', 24}, {'L', 25}, {'O', 26}, {'N', 27}, {'Q', 28}, {'P', 29}, {'S', 30}, {'R', 31}, {'U', 32}, {'T', 33}, {'W', 34}, {'V', 35}, {'Y', 36}, {'X', 37}, {'Z', 38}, {'a', 39}, {'c', 40}, {'b', 41}, {'e', 42}, {'d', 43}, {'g', 44}, {'f', 45}, {'i', 46}, {'h', 47}, {'k', 48}, {'j', 49}, {'m', 50}, {'l', 51}, {'o', 52}, {'n', 53}, {'q', 54}, {'p', 55}, {'s', 56}, {'r', 57}, {'u', 58}, {'t', 59}, {'w', 60}, {'v', 61}, {'y', 62}, {'x', 63}, {'z', 64}};

    // A mapping from index to character used by the tensorflow model.
    const std::vector<char> idToChar{{'\n', '!', ' ', '$', '\'', '&', '-', ',',
                                      '.', '3', ';', ':', '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K',
                                      'J', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X',
                                      'Z', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o',
                                      'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'}};
};

struct SampleCharRNNParams : samplesCommon::SampleParams
{
    int layerCount;
    int hiddenSize;
    int seqSize;
    int dataSize;
    int vocabSize;
    int outputSize;
    std::string weightFileName;

    SampleCharRNNMaps charMaps;
    SampleCharRNNWeightNames weightNames;
    SampleCharRNNBindingNames bindingNames;

    vector<std::string> inputSentences;
    vector<std::string> outputSentences;
};

//!
//! \brief  The SampleCharRNN class implements the char_rnn sample
//!
//! \details It uses weights from a trained TensorFlow model and creates
//!          the network using the TensorRT network definition API
//!
class SampleCharRNN
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleCharRNN(const SampleCharRNNParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

private:
    //!
    //! \brief Load requested weights from a formatted file into a map.
    //!
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

    //!
    //! \brief Converts RNN weights from TensorFlow's format to TensorRT's format.
    //!
    nvinfer1::Weights convertRNNWeights(nvinfer1::Weights input);

    //!
    //! \brief Converts RNN Biases from TensorFlow's format to TensorRT's format.
    //!
    nvinfer1::Weights convertRNNBias(nvinfer1::Weights input);

    //!
    //! \brief Add inputs to the TensorRT network and configure the RNNv2 layer using network definition API.
    //!
    nvinfer1::IRNNv2Layer* addRNNv2Layer(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief Create full model using the TensorRT network definition API and build the engine.
    //!
    void constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Looks up the embedding tensor for a given char and copies it to input buffer
    //!
    void copyEmbeddingToInput(samplesCommon::BufferManager& buffers, const char& c);

    //!
    //! \brief Perform one time step of inference with the TensorRT execution context
    //!
    bool stepOnce(samplesCommon::BufferManager& buffers,
                  SampleUniquePtr<nvinfer1::IExecutionContext>& context, cudaStream_t& stream);

    //!
    //! \brief Copies Ct/Ht output from the RNN to the Ct-1/Ht-1 input buffers for next time step
    //!
    void copyRNNOutputsToInputs(samplesCommon::BufferManager& buffers);

    std::map<std::string, nvinfer1::Weights> mWeightMap;
    SampleCharRNNParams mParams;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network
};

//!
//! \brief Creates the network, configures the builder and creates
//!        the network engine
//!
//! \details This function loads weights from a trained TensorFlow model,
//!          creates the network using the TensorRT network definition API,
//!          and builds a TensorRT engine.
//!
//! \return Returns true if the engine was created successfully and false
//!         otherwise
//!
bool SampleCharRNN::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    mWeightMap = SampleCharRNN::loadWeights(mParams.weightFileName);

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(32_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    constructNetwork(builder, network, config);

    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Load requested weights from a formatted file into a map.
//!
//! \param file Path to weights file. File has to be the formatted dump from
//!        the dumpTFWts.py script. Otherwise, this function will not work as
//!        intended.
//!
//! \return A map containing the extracted weights.
//!
//! \note  Weight V2 files are in a very simple space delimited format.
//!        <number of buffers>
//!        for each buffer: [name] [type] [shape] <data as binary blob>\n
//!        Note: type is the integer value of the DataType enum in NvInfer.h.
//!
std::map<std::string, nvinfer1::Weights> SampleCharRNN::loadWeights(const std::string file)
{
    std::map<std::string, nvinfer1::Weights> weightMap;

    std::ifstream input(file, std::ios_base::binary);
    assert(input.is_open() && "Unable to load weight file.");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        if (mParams.weightNames.names.empty())
        {
            break;
        }

        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // parse name and DataType
        std::string name;
        uint32_t type;
        input >> name >> std::dec >> type;
        wt.type = static_cast<nvinfer1::DataType>(type);

        // extract shape
        std::string temp, shape;
        std::getline(std::getline(input, temp, '('), shape, ')');

        // calculate count based on shape
        wt.count = 1;
        std::istringstream shapeStream(shape);
        while (std::getline(shapeStream, temp, ','))
            wt.count *= std::stoul(temp);
        size_t numOfBytes = samplesCommon::getElementSize(wt.type) * wt.count;

        // skip reading of weights if name is not in the set of names requested for extraction
        if (mParams.weightNames.names.find(name) == mParams.weightNames.names.end())
        {
            input.seekg(input.tellg() + static_cast<std::streamoff>(2 + numOfBytes));
            continue;
        }
        else
        {
            mParams.weightNames.names.erase(name);
        }

        // Read weight values
        input.seekg(input.tellg() + static_cast<std::streamoff>(1)); // skip space char
        char* wtVals = new char[numOfBytes];
        input.read(wtVals, numOfBytes);
        input.seekg(input.tellg() + static_cast<std::streamoff>(1)); // skip new-line char
        wt.values = wtVals;

        weightMap[name] = wt;
    }

    input.close();
    gLogInfo << "Done reading weights from file..." << std::endl;
    return weightMap;
}

//!
//! \brief Converts RNN weights from TensorFlow's format to TensorRT's format.
//!
//! \param input Weights that are stored in TensorFlow's format.
//!
//! \return Converted weights in TensorRT's format.
//!
//! \note TensorFlow weight parameters for BasicLSTMCell are formatted as:
//!       Each [WR][icfo] is hiddenSize sequential elements.
//!       CellN  Row 0: WiT, WcT, WfT, WoT
//!       CellN  Row 1: WiT, WcT, WfT, WoT
//!       ...
//!       CellN RowM-1: WiT, WcT, WfT, WoT
//!       CellN RowM+0: RiT, RcT, RfT, RoT
//!       CellN RowM+1: RiT, RcT, RfT, RoT
//!       ...
//!       CellNRow2M-1: RiT, RcT, RfT, RoT
//!
//!       TensorRT expects the format to laid out in memory:
//!       CellN: Wi, Wc, Wf, Wo, Ri, Rc, Rf, Ro
//!
nvinfer1::Weights SampleCharRNN::convertRNNWeights(nvinfer1::Weights input)
{
    float* ptr = new float[input.count];
    int dims[4]{2, mParams.hiddenSize, 4, mParams.hiddenSize};
    int order[4]{0, 3, 1, 2};
    utils::reshapeWeights(input, dims, order, ptr, 4);
    utils::transposeSubBuffers(ptr, DataType::kFLOAT, 2, mParams.hiddenSize * mParams.hiddenSize, 4);
    return nvinfer1::Weights{input.type, ptr, input.count};
}

//!
//! \brief Converts RNN Biases from TensorFlow's format to TensorRT's format.
//!
//! \param input Biases that are stored in TensorFlow's format.
//!
//! \return Converted bias in TensorRT's format.
//!
//! \note TensorFlow bias parameters for BasicLSTMCell are formatted as:
//!       CellN: Bi, Bc, Bf, Bo
//!
//!       TensorRT expects the format to be:
//!       CellN: Wi, Wc, Wf, Wo, Ri, Rc, Rf, Ro
//!
//!       Since tensorflow already combines U and W,
//!       we double the size and set all of U to zero.
nvinfer1::Weights SampleCharRNN::convertRNNBias(nvinfer1::Weights input)
{
    const int sizeOfElement = samplesCommon::getElementSize(input.type);
    char* ptr = new char[sizeOfElement * input.count * 2];
    const char* iptr = static_cast<const char*>(input.values);
    std::copy(iptr, iptr + 4 * mParams.hiddenSize * sizeOfElement, ptr);
    std::fill(ptr + sizeOfElement * input.count, ptr + sizeOfElement * input.count * 2, 0);
    return nvinfer1::Weights{input.type, ptr, input.count * 2};
}

//!
//! \brief Add inputs to the network and configure the RNNv2 layer using network definition API.
//!
//! \param network The network that will be used to build the engine.
//! \param weightMap Map that contains all the weights required by the model.
//!
//! \return Configured and added RNNv2 layer.
//!
nvinfer1::IRNNv2Layer* SampleCharRNN::addRNNv2Layer(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // Initialize data, hiddenIn, cellIn, and seqLenIn inputs into RNN Layer
    nvinfer1::ITensor* data = network->addInput(mParams.bindingNames.INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(mParams.seqSize, mParams.dataSize));
    assert(data != nullptr);

    nvinfer1::ITensor* hiddenIn = network->addInput(mParams.bindingNames.HIDDEN_IN_BLOB_NAME, nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(mParams.layerCount, mParams.hiddenSize));
    assert(hiddenIn != nullptr);

    nvinfer1::ITensor* cellIn = network->addInput(mParams.bindingNames.CELL_IN_BLOB_NAME, nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(mParams.layerCount, mParams.hiddenSize));
    assert(cellIn != nullptr);

    nvinfer1::ITensor* seqLenIn = network->addInput(mParams.bindingNames.SEQ_LEN_IN_BLOB_NAME, nvinfer1::DataType::kINT32, nvinfer1::Dims{});
    assert(seqLenIn != nullptr);

    // create an RNN layer w/ 2 layers and 512 hidden states
    nvinfer1::IRNNv2Layer* rnn = network->addRNNv2(*data, mParams.layerCount, mParams.hiddenSize, mParams.seqSize, nvinfer1::RNNOperation::kLSTM);
    assert(rnn != nullptr);

    // Set RNNv2 optional inputs
    rnn->getOutput(0)->setName("RNN output");
    rnn->setHiddenState(*hiddenIn);
    if (rnn->getOperation() == nvinfer1::RNNOperation::kLSTM)
    {
        rnn->setCellState(*cellIn);
    }

    // Specify sequence lengths.  Note this can be omitted since we are always using the maximum
    // sequence length, but for illustrative purposes we explicitly pass in sequence length data
    // in the sample
    rnn->setSequenceLengths(*seqLenIn);
    seqLenIn->setLocation(nvinfer1::TensorLocation::kDEVICE);

    // convert tensorflow weight format to trt weight format
    nvinfer1::Weights rnnwL0 = SampleCharRNN::convertRNNWeights(mWeightMap[mParams.weightNames.RNNW_L0_NAME]);
    nvinfer1::Weights rnnbL0 = SampleCharRNN::convertRNNBias(mWeightMap[mParams.weightNames.RNNB_L0_NAME]);
    nvinfer1::Weights rnnwL1 = SampleCharRNN::convertRNNWeights(mWeightMap[mParams.weightNames.RNNW_L1_NAME]);
    nvinfer1::Weights rnnbL1 = SampleCharRNN::convertRNNBias(mWeightMap[mParams.weightNames.RNNB_L1_NAME]);

    std::vector<nvinfer1::RNNGateType> gateOrder({nvinfer1::RNNGateType::kINPUT,
                                                  nvinfer1::RNNGateType::kCELL,
                                                  nvinfer1::RNNGateType::kFORGET,
                                                  nvinfer1::RNNGateType::kOUTPUT});
    const nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(rnnwL0.type);
    const float* wtsL0 = static_cast<const float*>(rnnwL0.values);
    const float* biasesL0 = static_cast<const float*>(rnnbL0.values);
    const float* wtsL1 = static_cast<const float*>(rnnwL1.values);
    const float* biasesL1 = static_cast<const float*>(rnnbL1.values);
    size_t kernelOffset = 0, biasOffset = 0;
    for (int gateIndex = 0, numGates = gateOrder.size(); gateIndex < 2 * numGates; gateIndex++)
    {
        // extract weights and bias for a given gate and layer
        nvinfer1::Weights gateWeightL0{dataType, wtsL0 + kernelOffset, mParams.dataSize * mParams.hiddenSize};
        nvinfer1::Weights gateBiasL0{dataType, biasesL0 + biasOffset, mParams.hiddenSize};
        nvinfer1::Weights gateWeightL1{dataType, wtsL1 + kernelOffset, mParams.dataSize * mParams.hiddenSize};
        nvinfer1::Weights gateBiasL1{dataType, biasesL1 + biasOffset, mParams.hiddenSize};

        // set weights and bias for given gate
        rnn->setWeightsForGate(0, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateWeightL0);
        rnn->setBiasForGate(0, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateBiasL0);
        rnn->setWeightsForGate(1, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateWeightL1);
        rnn->setBiasForGate(1, gateOrder[gateIndex % numGates], (gateIndex < numGates), gateBiasL1);

        // Update offsets
        kernelOffset = kernelOffset + mParams.dataSize * mParams.hiddenSize;
        biasOffset = biasOffset + mParams.hiddenSize;
    }

    // Store the transformed weights in the weight map so the memory can be properly released later.
    mWeightMap["rnnwL0"] = rnnwL0;
    mWeightMap["rnnbL0"] = rnnbL0;
    mWeightMap["rnnwL1"] = rnnwL1;
    mWeightMap["rnnbL1"] = rnnbL1;

    return rnn;
}

//!
//! \brief Create full model using the TensorRT network definition API and build the engine.
//!
//! \param weightMap Map that contains all the weights required by the model.
//! \param modelStream The stream within which the engine is serialized once built.
//!
void SampleCharRNN::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    // add RNNv2 layer and set its parameters
    auto rnn = SampleCharRNN::addRNNv2Layer(network);

    // Transpose FC weights since TensorFlow's weights are transposed when compared to TensorRT
    utils::transposeSubBuffers((void*) mWeightMap[mParams.weightNames.FCW_NAME].values, nvinfer1::DataType::kFLOAT, 1, mParams.hiddenSize, mParams.vocabSize);

    // add Constant layers for fully connected weights
    auto fcwts = network->addConstant(nvinfer1::Dims2(mParams.vocabSize, mParams.hiddenSize), mWeightMap[mParams.weightNames.FCW_NAME]);

    // Add matrix multiplication layer for multiplying rnn output with FC weights
    auto matrixMultLayer = network->addMatrixMultiply(*fcwts->getOutput(0), false, *rnn->getOutput(0), true);
    assert(matrixMultLayer != nullptr);
    matrixMultLayer->getOutput(0)->setName("Matrix Multiplicaton output");

    // Add elementwise layer for adding bias
    auto fcbias = network->addConstant(nvinfer1::Dims2(mParams.vocabSize, 1), mWeightMap[mParams.weightNames.FCB_NAME]);
    auto addBiasLayer = network->addElementWise(*matrixMultLayer->getOutput(0), *fcbias->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer != nullptr);
    addBiasLayer->getOutput(0)->setName("Add Bias output");

    // Add TopK layer to determine which character has highest probability.
    int reduceAxis = 0x1; // reduce across vocab axis
    auto pred = network->addTopK(*addBiasLayer->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, reduceAxis);
    assert(pred != nullptr);
    pred->getOutput(1)->setName(mParams.bindingNames.OUTPUT_BLOB_NAME);

    // Mark the outputs for the network
    network->markOutput(*pred->getOutput(1));
    pred->getOutput(1)->setType(nvinfer1::DataType::kINT32);
    rnn->getOutput(1)->setName(mParams.bindingNames.HIDDEN_OUT_BLOB_NAME);
    network->markOutput(*rnn->getOutput(1));
    if (rnn->getOperation() == nvinfer1::RNNOperation::kLSTM)
    {
        rnn->getOutput(2)->setName(mParams.bindingNames.CELL_OUT_BLOB_NAME);
        network->markOutput(*rnn->getOutput(2));
    }

    gLogInfo << "Done constructing network..." << std::endl;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It
//!          allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleCharRNN::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());

    if (!context)
    {
        return false;
    }

    // Select a random seed string.
    srand(unsigned(time(nullptr)));
    int sentenceIndex = rand() % mParams.inputSentences.size();
    std::string inputSentence = mParams.inputSentences[sentenceIndex];
    std::string expected = mParams.outputSentences[sentenceIndex];
    std::string genstr;

    gLogInfo << "RNN warmup sentence: " << inputSentence << std::endl;
    gLogInfo << "Expected output: " << expected << std::endl;

    // create stream for trt execution
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Set sequence lengths to maximum
    std::fill_n(reinterpret_cast<int32_t*>(buffers.getHostBuffer(mParams.bindingNames.SEQ_LEN_IN_BLOB_NAME)), mParams.batchSize, mParams.seqSize);

    // Initialize hiddenIn and cellIn tensors to zero before seeding
    void* hiddenIn = buffers.getHostBuffer(mParams.bindingNames.HIDDEN_IN_BLOB_NAME);
    auto hiddenTensorSize = buffers.size(mParams.bindingNames.HIDDEN_IN_BLOB_NAME);

    void* cellIn = buffers.getHostBuffer(mParams.bindingNames.CELL_IN_BLOB_NAME);
    auto cellTensorSize = buffers.size(mParams.bindingNames.CELL_IN_BLOB_NAME);

    std::memset(hiddenIn, 0, hiddenTensorSize);
    std::memset(cellIn, 0, cellTensorSize);

    // Seed the RNN with the input sentence.
    for (auto& a : inputSentence)
    {
        SampleCharRNN::copyEmbeddingToInput(buffers, a);

        if (!SampleCharRNN::stepOnce(buffers, context, stream))
        {
            return false;
        }

        SampleCharRNN::copyRNNOutputsToInputs(buffers);
        genstr.push_back(a);
    }

    // Extract first predicted character
    uint32_t predIdx = *reinterpret_cast<uint32_t*>(buffers.getHostBuffer(mParams.bindingNames.OUTPUT_BLOB_NAME));
    genstr.push_back(mParams.charMaps.idToChar.at(predIdx));

    // Generate predicted sequence of characters
    for (size_t x = 0, y = expected.size() - 1; x < y; x++)
    {
        SampleCharRNN::copyEmbeddingToInput(buffers, *genstr.rbegin());

        if (!SampleCharRNN::stepOnce(buffers, context, stream))
        {
            return false;
        }

        SampleCharRNN::copyRNNOutputsToInputs(buffers);
        predIdx = *reinterpret_cast<uint32_t*>(buffers.getHostBuffer(mParams.bindingNames.OUTPUT_BLOB_NAME));
        genstr.push_back(mParams.charMaps.idToChar.at(predIdx));
    }

    gLogInfo << "Received: " << genstr.substr(inputSentence.size()) << std::endl;

    // release the stream
    cudaStreamDestroy(stream);

    return genstr == (inputSentence + expected);
}

//!
//! \brief Looks up the embedding tensor for a given char and copies it to input buffer
//!
void SampleCharRNN::copyEmbeddingToInput(samplesCommon::BufferManager& buffers, const char& c)
{
    auto embed = mWeightMap[mParams.weightNames.EMBED_NAME];
    float* inputBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.bindingNames.INPUT_BLOB_NAME));
    auto index = mParams.charMaps.charToID.at(c);

    std::memcpy(inputBuffer, static_cast<const float*>(embed.values) + index * mParams.dataSize, buffers.size(mParams.bindingNames.INPUT_BLOB_NAME));
}

//!
//! \brief Perform one time step of inference with the TensorRT execution context
//!
bool SampleCharRNN::stepOnce(samplesCommon::BufferManager& buffers, SampleUniquePtr<nvinfer1::IExecutionContext>& context, cudaStream_t& stream)
{
    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return false;
    }
    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    cudaStreamSynchronize(stream);
    return true;
}

//!
//! \brief Copies Ct/Ht output from the RNN to the Ct-1/Ht-1 input buffers for next time step
//!
void SampleCharRNN::copyRNNOutputsToInputs(samplesCommon::BufferManager& buffers)
{
    // Copy Ct/Ht to the Ct-1/Ht-1 slots.
    void* hiddenIn = buffers.getHostBuffer(mParams.bindingNames.HIDDEN_IN_BLOB_NAME);
    void* hiddenOut = buffers.getHostBuffer(mParams.bindingNames.HIDDEN_OUT_BLOB_NAME);
    auto hiddenTensorSize = buffers.size(mParams.bindingNames.HIDDEN_IN_BLOB_NAME);

    void* cellIn = buffers.getHostBuffer(mParams.bindingNames.CELL_IN_BLOB_NAME);
    void* cellOut = buffers.getHostBuffer(mParams.bindingNames.CELL_OUT_BLOB_NAME);
    auto cellTensorSize = buffers.size(mParams.bindingNames.CELL_IN_BLOB_NAME);

    std::memcpy(hiddenIn, hiddenOut, hiddenTensorSize);
    std::memcpy(cellIn, cellOut, cellTensorSize);
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleCharRNN::teardown()
{
    // Clean up runtime resources
    for (auto& mem : mWeightMap)
    {
        delete[] static_cast<const float*>(mem.second.values);
    }

    return true;
}

//!
//! \brief Initializes members of the params struct using the
//!        command line args
//!
SampleCharRNNParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleCharRNNParams params;

    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("data/char-rnn/");
        params.dataDirs.push_back("data/samples/char-rnn/");
    }
    else
    {
        params.dataDirs = args.dataDirs;
    }

    params.batchSize = 1;
    params.layerCount = 2;
    params.hiddenSize = 512;
    params.seqSize = 1;
    params.dataSize = params.hiddenSize;
    params.vocabSize = 65;
    params.outputSize = 1;
    params.weightFileName = locateFile("char-rnn.wts", params.dataDirs);

    // Input strings and their respective expected output strings
    const std::vector<std::string> inS{
        "ROMEO",
        "JUL",
        "The K",
        "That tho",
        "KING",
        "beauty of",
        "birth of K",
        "Hi",
        "JACK",
        "interestingly, it was J",
    };
    const std::vector<std::string> outS{
        ":\nThe sense to",
        "IET:\nWhat shall I shall be",
        "ing Richard shall be the strange",
        "u shalt be the",
        " HENRY VI:\nWhat",
        " the son,",
        "ing Richard's son",
        "ng of York,\nThat thou hast so the",
        "INGHAM:\nWhat shall I",
        "uliet",
    };

    params.inputSentences = inS;
    params.outputSentences = outS;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_char_rnn [-h or --help] [-d or --datadir=<path to data directory>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use data/samples/char-rnn/ and data/char-rnn/" << std::endl;
}

//!
//! \brief Runs the char-rnn model in TensorRT with a set of expected input and output strings.
//!
int main(int argc, char** argv)
{
    setReportableSeverity(Logger::Severity::kVERBOSE);
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleCharRNNParams params = initializeSampleParams(args);
    SampleCharRNN sample(params);

    gLogInfo << "Building and running a GPU inference engine for Char RNN model..."
             << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}

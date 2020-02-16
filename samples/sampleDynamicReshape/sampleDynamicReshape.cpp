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
//! sampleDynamicReshape.cpp
//! This file contains the implementation of the dynamic reshape MNIST sample. It creates a network
//! using the MNIST ONNX model, and uses a second engine to resize inputs to the shape the model
//! expects.
//! It can be run with the following command:
//! Command: ./sample_dynamic_reshape [-h or --help [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <random>

const std::string gSampleName = "TensorRT.sample_dynamic_reshape";

//! \brief The SampleDynamicReshape class implementes the dynamic reshape sample.
//!
//! \details This class builds one engine that resizes a given input to the correct size, and a
//! second engine based on an ONNX MNIST model that generates a prediction.
//!
class SampleDynamicReshape
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleDynamicReshape(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds both engines.
    //!
    void build();

    //!
    //! \brief Prepares the model for inference by creating execution contexts and allocating buffers.
    //!
    void prepare();

    //!
    //! \brief Runs inference using TensorRT on a random image.
    //!
    bool infer();

private:
    void buildPreprocessorEngine(const SampleUniquePtr<nvinfer1::IBuilder>& builder);
    void buildPredictionEngine(const SampleUniquePtr<nvinfer1::IBuilder>& builder);

    Dims loadPGMFile(const std::string& fileName);
    bool validateOutput(int digit);

    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mPredictionInputDims; //!< The dimensions of the input of the MNIST model.
    nvinfer1::Dims mPredicitionOutputDims; //!< The dimensions of the output of the MNIST model.

    // Engines used for inference. The first is used for resizing inputs, the second for prediction.
    SampleUniquePtr<nvinfer1::ICudaEngine> mPreprocessorEngine{nullptr}, mPredictionEngine{nullptr};

    SampleUniquePtr<nvinfer1::IExecutionContext> mPreprocessorContext{nullptr}, mPredictionContext{nullptr};

    samplesCommon::ManagedBuffer mInput{}; //!< Host and device buffers for the input.
    samplesCommon::DeviceBuffer mPredictionInput{}; //!< Device buffer for the output of the preprocessor, i.e. the input to the prediction model.
    samplesCommon::ManagedBuffer mOutput{}; //!< Host buffer for the ouptut

    template <typename T>
    SampleUniquePtr<T> makeUnique(T* t)
    {
        if (!t)
        {
            throw std::runtime_error{"Failed to create TensorRT object"};
        }
        return SampleUniquePtr<T>{t};
    }
};

//!
//! \brief Builds the two engines required for inference.
//!
//! \details This function creates one TensorRT engine for resizing inputs to the correct sizes,
//!          then creates a TensorRT network by parsing the ONNX model and builds
//!          an engine that will be used to run inference (mPredictionEngine).
//!
void SampleDynamicReshape::build()
{
    auto builder = this->makeUnique(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));

    // This function will also set mPredictionInputDims and mPredicitionOutputDims,
    // so it needs to be called before building the preprocessor.
    this->buildPredictionEngine(builder);
    this->buildPreprocessorEngine(builder);
}

//!
//! \brief Builds an engine for preprocessing (mPreprocessorEngine).
//!
void SampleDynamicReshape::buildPreprocessorEngine(const SampleUniquePtr<nvinfer1::IBuilder>& builder)
{
    // Create the preprocessor engine using a network that supports full dimensions (createNetworkV2).
    auto preprocessorNetwork = this->makeUnique(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

    // Reshape a dynamically shaped input to the size expected by the model, (1, 28, 28).
    auto input = preprocessorNetwork->addInput("input", nvinfer1::DataType::kFLOAT, Dims3{1, -1, -1});
    auto resizeLayer = preprocessorNetwork->addResize(*input);
    resizeLayer->setOutputDimensions(mPredictionInputDims);
    preprocessorNetwork->markOutput(*resizeLayer->getOutput(0));

    // Finally, configure and build the preprocessor engine.
    auto preprocessorConfig = this->makeUnique(builder->createBuilderConfig());

    // Create an optimization profile so that we can specify a range of input dimensions.
    auto profile = builder->createOptimizationProfile();

    // This profile will be valid for all images whose size falls in the range of [(1, 1, 1), (1, 56, 56)]
    // but the TensorRT will optimize for (1, 28, 28)
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims3{1, 1, 1});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims3{1, 28, 28});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims3{1, 56, 56});
    preprocessorConfig->addOptimizationProfile(profile);
    mPreprocessorEngine = this->makeUnique(builder->buildEngineWithConfig(*preprocessorNetwork, *preprocessorConfig));
    gLogInfo << "Profile dimensions in preprocessor engine:\n";
    gLogInfo << "    Minimum = " << mPreprocessorEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN) << '\n';
    gLogInfo << "    Optimum = " << mPreprocessorEngine->getProfileDimensions(0, 0, OptProfileSelector::kOPT) << '\n';
    gLogInfo << "    Maximum = " << mPreprocessorEngine->getProfileDimensions(0, 0, OptProfileSelector::kMAX) << std::endl;
}

//!
//! \brief Builds an engine for prediction (mPredictionEngine).
//!
//! \details This function builds an engine for the MNIST model, and updates mPredictionInputDims and mPredicitionOutputDims
//! according to the dimensions specified by the model. The preprocessor reshapes inputs to mPredictionInputDims.
//!
void SampleDynamicReshape::buildPredictionEngine(const SampleUniquePtr<nvinfer1::IBuilder>& builder)
{
    // Create a network using the parser.
    auto network = this->makeUnique(builder->createNetwork());
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    bool parsingSuccess = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsingSuccess)
    {
        throw std::runtime_error{"Failed to parse model"};
    }

    // Attach a softmax layer to the end of the network.
    auto softmax = network->addSoftMax(*network->getOutput(0));
    network->unmarkOutput(*network->getOutput(0));
    network->markOutput(*softmax->getOutput(0));

    // Get information about the inputs/outputs directly from the model.
    mPredictionInputDims = network->getInput(0)->getDimensions();
    mPredicitionOutputDims = network->getOutput(0)->getDimensions();

    // Create a builder config
    auto config = this->makeUnique(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    // Build the prediciton engine.
    mPredictionEngine = this->makeUnique(builder->buildEngineWithConfig(*network, *config));
}

//!
//! \brief Prepares the model for inference by creating an execution context and allocating buffers.
//!
//! \details This function sets up the sample for inference. This involves allocating buffers for the inputs and outputs,
//! as well as creating TensorRT execution contexts for both engines. This only needs to be called a single time.
//!
void SampleDynamicReshape::prepare()
{
    mPreprocessorContext = this->makeUnique(mPreprocessorEngine->createExecutionContext());
    mPredictionContext = this->makeUnique(mPredictionEngine->createExecutionContext());
    // Since input dimensions are not known ahead of time, we only allocate the output buffer and preprocessor output buffer.
    mPredictionInput.resize(mPredictionInputDims);
    mOutput.hostBuffer.resize(mPredicitionOutputDims);
    mOutput.deviceBuffer.resize(mPredicitionOutputDims);
}

//!
//! \brief Runs inference for this sample
//!
//! \details This function is the main execution function of the sample.
//! It runs inference for using a random image from the MNIST dataset as an input.
//!
bool SampleDynamicReshape::infer()
{
    // Load a random PGM file into a host buffer, then copy to device.
    std::random_device rd{};
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int> digitDistribution{0, 9};
    int digit = digitDistribution(generator);

    Dims inputDims = this->loadPGMFile(locateFile(std::to_string(digit) + ".pgm", mParams.dataDirs));
    mInput.deviceBuffer.resize(inputDims);
    CHECK(cudaMemcpy(mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

    // Set the input size for the preprocessor
    mPreprocessorContext->setBindingDimensions(0, inputDims);
    // We can only run inference once all dynamic input shapes have been specified.
    if (!mPreprocessorContext->allInputDimensionsSpecified())
    {
        return false;
    }

    // Run the preprocessor to resize the input to the correct shape
    std::vector<void*> preprocessorBindings = {mInput.deviceBuffer.data(), mPredictionInput.data()};
    // For engines using full dims, we can use executeV2, which does not include a separate batch size parameter.
    bool status = mPreprocessorContext->executeV2(preprocessorBindings.data());
    if (!status)
    {
        return false;
    }

    // Next, run the model to generate a prediction.
    std::vector<void*> predicitonBindings = {mPredictionInput.data(), mOutput.deviceBuffer.data()};
    status = mPredictionContext->execute(mParams.batchSize, predicitonBindings.data());
    if (!status)
    {
        return false;
    }

    // Copy the outputs back to the host and verify the output.
    CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
    return validateOutput(digit);
}

//!
//! \brief Loads a PGM file into mInput and returns the dimensions of the loaded image.
//!
//! \details This function loads the specified PGM file into the input host buffer.
//!
Dims SampleDynamicReshape::loadPGMFile(const std::string& fileName)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");

    std::string magic;
    int h, w, max;
    infile >> magic >> h >> w >> max;

    infile.seekg(1, infile.cur);
    Dims3 inputDims{1, h, w};
    size_t vol = samplesCommon::volume(inputDims);
    std::vector<uint8_t> fileData(vol);
    infile.read(reinterpret_cast<char*>(fileData.data()), vol);

    // Print an ascii representation
    gLogInfo << "Input:\n";
    for (size_t i = 0; i < vol; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % w) ? "" : "\n");
    }
    gLogInfo << std::endl;

    // Normalize and copy to the host buffer.
    mInput.hostBuffer.resize(inputDims);
    float* hostDataBuffer = static_cast<float*>(mInput.hostBuffer.data());
    std::transform(fileData.begin(), fileData.end(), hostDataBuffer, [] (uint8_t x) { return 1.0 - static_cast<float>(x / 255.0); });
    return inputDims;
}

//!
//! \brief Checks whether the model prediction (in mOutput) is correct.
//!
bool SampleDynamicReshape::validateOutput(int digit)
{
    const float* bufRaw = static_cast<const float*>(mOutput.hostBuffer.data());
    std::vector<float> prob(bufRaw, bufRaw + mOutput.hostBuffer.size());

    int curIndex{0};
    for (const auto& elem : prob)
    {
        gLogInfo << " Prob " << curIndex << "  " << std::fixed << std::setw(5) << std::setprecision(4) << elem << " "
                 << "Class " << curIndex << ": " << std::string(int(std::floor(elem * 10 + 0.5f)), '*') << std::endl;
        ++curIndex;
    }

    int predictedDigit = std::max_element(prob.begin(), prob.end()) - prob.begin();
    return digit == predictedDigit;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.batchSize = 1;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_dynamic_reshape [-h or --help] [-d or --datadir=<path to data directory>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
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

    SampleDynamicReshape sample{initializeSampleParams(args)};

    sample.build();
    sample.prepare();

    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}

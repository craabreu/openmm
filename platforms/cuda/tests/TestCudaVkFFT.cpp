/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * This tests the CUDA implementation of FFT.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "fftpack.h"
#include "sfmt/SFMT.h"
#include "openmm/System.h"
#include <string>

#define VKFFT_BACKEND 1 // CUDA
#include "openmm/common/vkFFT.h"

using namespace OpenMM;
using namespace std;

static CudaPlatform platform;

template <class Real2>
void testTransform(bool realToComplex, int xsize, int ysize, int zsize) {
    System system;
    system.addParticle(0.0);
    CudaPlatform::PlatformData platformData(NULL, system, "", "true", platform.getPropertyDefaultValue("CudaPrecision"), "false",
            platform.getPropertyDefaultValue(CudaPlatform::CudaCompiler()), platform.getPropertyDefaultValue(CudaPlatform::CudaTempDirectory()),
            platform.getPropertyDefaultValue(CudaPlatform::CudaHostCompiler()), platform.getPropertyDefaultValue(CudaPlatform::CudaDisablePmeStream()), "false", true, 1, NULL);
    CudaContext& context = *platformData.contexts[0];
    context.initialize();
    context.setAsCurrent();
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Real2> original(xsize*ysize*zsize);
    vector<t_complex> reference(original.size());
    for (int i = 0; i < (int) original.size(); i++) {
        Real2 value;
        value.x = (float) genrand_real2(sfmt);
        value.y = (float) genrand_real2(sfmt);
        original[i] = value;
        reference[i] = t_complex(value.x, value.y);
    }
    for (int i = 0; i < (int) reference.size(); i++) {
        if (realToComplex)
            reference[i] = t_complex(i%2 == 0 ? original[i/2].x : original[i/2].y, 0);
        else
            reference[i] = t_complex(original[i].x, original[i].y);
    }
    CudaArray grid1(context, original.size(), sizeof(Real2), "grid1");
    CudaArray grid2(context, original.size(), sizeof(Real2), "grid2");
    grid1.upload(original);

    int device = context.getDeviceIndex();
    CUstream stream = context.getCurrentStream();
    void* inputBuffer = (void*) grid1.getDevicePointer();
    void* outputBuffer = (void*) grid2.getDevicePointer();

    bool doublePrecision = typeid(Real2) == typeid(double2);
    int outputZSize = realToComplex ? (zsize/2+1) : zsize;
    size_t realTypeSize = doublePrecision ? sizeof(double) : sizeof(float);
    size_t inputElementSize = realToComplex ? realTypeSize : 2*realTypeSize;
    uint64_t inputBufferSize = inputElementSize*zsize*ysize*xsize;
    uint64_t outputBufferSize = 2*realTypeSize*outputZSize*ysize*xsize;

    VkFFTConfiguration config = {};

    config.performR2C = realToComplex;
    config.device = &device;
    config.num_streams = 1;
    config.stream = &stream;
    config.doublePrecision = doublePrecision;

    config.FFTdim = 3;
    config.size[0] = zsize;
    config.size[1] = ysize;
    config.size[2] = xsize;

    config.inverseReturnToInputBuffer = true;
    config.isInputFormatted = true;
    config.inputBufferSize = &inputBufferSize;
    config.inputBuffer = &inputBuffer;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = zsize*ysize;
    config.inputBufferStride[2] = zsize*ysize*xsize;

    config.bufferSize = &outputBufferSize;
    config.buffer = &outputBuffer;
    config.bufferStride[0] = outputZSize;
    config.bufferStride[1] = outputZSize*ysize;
    config.bufferStride[2] = outputZSize*ysize*xsize;

    VkFFTApplication* app = new VkFFTApplication();
    VkFFTResult status = initializeVkFFT(app, config);
    if (status != VKFFT_SUCCESS)
        throw OpenMMException("Error initializing VkFFT: "+to_string(status));

    // Perform a forward FFT, then verify the result is correct.

    status = VkFFTAppend(app, -1, NULL);
    if (status != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+to_string(status));

    vector<Real2> result;
    grid2.download(result);
    fftpack_t plan;
    fftpack_init_3d(&plan, xsize, ysize, zsize);
    fftpack_exec_3d(plan, FFTPACK_FORWARD, &reference[0], &reference[0]);
    for (int x = 0; x < xsize; x++)
        for (int y = 0; y < ysize; y++)
            for (int z = 0; z < outputZSize; z++) {
                int index1 = x*ysize*zsize + y*zsize + z;
                int index2 = x*ysize*outputZSize + y*outputZSize + z;
                ASSERT_EQUAL_TOL(reference[index1].re, result[index2].x, 1e-3);
                ASSERT_EQUAL_TOL(reference[index1].im, result[index2].y, 1e-3);
            }
    fftpack_destroy(plan);

    // Perform a backward transform and see if we get the original values.

    status = VkFFTAppend(app, 1, NULL);
    if (status != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+to_string(status));

    grid1.download(result);
    double scale = 1.0/(xsize*ysize*zsize);
    int valuesToCheck = (realToComplex ? original.size()/2 : original.size());
    for (int i = 0; i < valuesToCheck; ++i) {
        ASSERT_EQUAL_TOL(original[i].x, scale*result[i].x, 1e-4);
        ASSERT_EQUAL_TOL(original[i].y, scale*result[i].y, 1e-4);
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        if (platform.getPropertyDefaultValue("CudaPrecision") == "double") {
            testTransform<double2>(false, 28, 25, 30);
            testTransform<double2>(true, 28, 25, 25);
            testTransform<double2>(true, 25, 28, 25);
            testTransform<double2>(true, 25, 25, 28);
            testTransform<double2>(true, 21, 25, 27);
        }
        else {
            testTransform<float2>(false, 28, 25, 30);
            testTransform<float2>(true, 28, 25, 25);
            testTransform<float2>(true, 25, 28, 25);
            testTransform<float2>(true, 25, 25, 28);
            testTransform<float2>(true, 21, 25, 27);
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}

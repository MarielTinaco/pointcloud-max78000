/*******************************************************************************
* Copyright (C) 2019-2022 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// depthmap
// Created using ai8xize.py --test-dir sdk/Examples/MAX78000/CNN --prefix depthmap --checkpoint-file trained/ai85-depth-qat_best-q.pth.tar --config-file networks/depthmap.yaml --softmax --device MAX78000 --fifo --timer 0 --display-checkpoint --verbose --overwrite

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// Data input: CHW 1x128x128 (16384 bytes):
static const uint32_t bathtub[SAMPLE_SIZE_bathtub][4096] =  SAMPLE_INPUT_bathtub

static const uint32_t bed[SAMPLE_SIZE_bed][4096] =  SAMPLE_INPUT_bed

static const uint32_t chair[SAMPLE_SIZE_chair][4096] =  SAMPLE_INPUT_chair

static const uint32_t desk[SAMPLE_SIZE_desk][4096] =  SAMPLE_INPUT_desk

static const uint32_t dresser[SAMPLE_SIZE_dresser][4096] =  SAMPLE_INPUT_dresser

static const uint32_t monitor[SAMPLE_SIZE_monitor][4096] =  SAMPLE_INPUT_monitor

static const uint32_t nightstand[SAMPLE_SIZE_night_stand][4096] =  SAMPLE_INPUT__night_stand

static const uint32_t sofa[SAMPLE_SIZE_sofa][4096] =  SAMPLE_INPUT_sofa

static const uint32_t table[SAMPLE_SIZE_table][4096] =  SAMPLE_INPUT_table

static const uint32_t toilet[SAMPLE_SIZE_toilet][4096] =  SAMPLE_INPUT_toilet

static const uint32_t *input = {
  bathtub,
  bed,
  chair,
  desk,
  dresser,
  monitor,
  nightstand,
  sofa,
  table,
  toilet
}

static const uint32_t input_0 = input[0][0]

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  int i;
  const uint32_t *in0 = input_0;

  for (i = 0; i < 4096; i++) {
    // Remove the following line if there is no risk that the source would overrun the FIFO:
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
  }
}

// Expected output of layer 13 for depthmap given the sample input (known-answer test)
// Delete this function for production code
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{
  int i;
  int digs, tens;
  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: APB (50 MHz) div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test ***\n");

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  cnn_load_bias();
  cnn_configure(); // Configure state machine
  cnn_start(); // Start CNN processing
  load_input(); // Load data input via FIFO

  SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0
  while (cnn_time == 0)
    __WFI(); // Wait for CNN

  // if (check_output() != CNN_OK) fail();
  softmax_layer();

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate data loading and inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable(); // Shut down CNN clock, disable peripheral

  printf("Classification results:\n");
  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
    tens = digs % 10;
    digs = digs / 10;
    printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
  }

  return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 57,813,376 ops (56,861,952 macc; 951,424 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 2,621,440 ops (2,359,296 macc; 262,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 12,140,544 ops (11,796,480 macc; 344,064 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 14,827,520 ops (14,745,600 macc; 81,920 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 14,827,520 ops (14,745,600 macc; 81,920 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 3,788,800 ops (3,686,400 macc; 102,400 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 3,706,880 ops (3,686,400 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 2,059,264 ops (2,027,520 macc; 31,744 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 1,230,848 ops (1,216,512 macc; 14,336 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 1,330,176 ops (1,327,104 macc; 3,072 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 668,160 ops (663,552 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 200,192 ops (196,608 macc; 3,584 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 262,656 ops (262,144 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 148,096 ops (147,456 macc; 640 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 1,280 ops (1,280 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 369,984 bytes out of 442,368 bytes total (84%)
  Bias memory:   1,094 bytes out of 2,048 bytes total (53%)
*/


import { Matrix, Vector } from '../utils/math.js';
import DenseLayer from '../layers/denseLayer.js'
import NN from '../core/NN.js'
import { describe, it } from 'node:test';
import assert from 'assert';
import { mse } from '../utils/loss.js';

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Derivative of the sigmoid function
function sigmoidDerivative(x) {
    const sig = sigmoid(x);
    return sig * (1 - sig);
}

const sigmoidActivation = {
    func: sigmoid,
    derivative: sigmoidDerivative
};

const h = 0.00001;
const mseErrorFunction = {
    func: mse,
    derivative: (yTrue, yPred) => {
        const result = [];
        for (let i = 0; i < yTrue.length; i++) {
            const yPredH = [...yPred];
            yPredH[i] += h;
            result.push((mse(yTrue, yPredH) - mse(yTrue, yPred)) / h);
        }
        return result;
    }
};

describe('Matrix and Vector operations', () => {
    const v = (...values) => new Vector(...values);
    const m = new Matrix(v(3, -2), v(5, -4));
    const m2 = new Matrix(v(3, 4), v(2, 5));
    const m3 = new Matrix(v(1), v(2), v(3));
    const m4 = new Matrix(v(1, 2, 3));

    describe('Matrix multiplication', () => {
        it('should multiply matrix to number', () => {
            const mul = m.mul(2);
            assert.deepStrictEqual(mul, new Matrix(v(6, -4), v(10, -8)));
        })

        it('should multiply each element of matrix m by 3', () => {
            const mul = m.mul(3);
            assert.deepStrictEqual(mul, new Matrix(v(9, -6), v(15, -12)));
        });

        it('should multiply matrix m by matrix m2', () => {
            const mul2 = m.mul(m2);
            // This result depends on the implementation of matrix multiplication
            assert.deepStrictEqual(mul2, new Matrix(v(5, 2), v(7, 0)));
        });

        it('should multiply matrix m2 by matrix m', () => {
            const mul3 = m2.mul(m);
            // This result depends on the implementation of matrix multiplication
            assert.deepStrictEqual(mul3, new Matrix(v(29, -22), v(31, -24)));
        });

        it('should multiply matrix m3 by matrix m4', () => {
            const mul4 = m3.mul(m4);
            // Verify based on matrix multiplication rules
            assert.deepStrictEqual(mul4, new Matrix(v(1, 2, 3), v(2, 4, 6), v(3, 6, 9)));
        });

        it('should multiply matrix m4 by matrix m3', () => {
            const mul5 = m4.mul(m3);
            // Verify based on matrix multiplication rules
            assert.deepStrictEqual(mul5, new Matrix(v(14)));
        });
    });

    describe('Matrix addition', () => {
        it('should add matrix m and matrix m2', () => {
            const add = m.add(m2);
            assert.deepStrictEqual(add, new Matrix(v(6, 2), v(7, 1)));
        });
    });

    describe('Matrix transpose', () => {
        it('should transpose matrix m4', () => {
            const m4T = m4.transpose();
            assert.deepStrictEqual(m4T, new Matrix(v(1), v(2), v(3)));
        });

        it('should multiply matrix m4 by its transpose', () => {
            const m4T = m4.transpose();
            const prod = m4.mul(m4T);
            assert.deepStrictEqual(prod, new Matrix(v(14)));  // Assuming m4.mul calculates dot product for 1x3 and 3x1 matrices
        });
    });
});

describe('DenseLayer Class', () => {
    describe('Initialization', () => {
        it('should initialize weights and biases correctly', () => {
            const layer = new DenseLayer(2, 2);
            assert.strictEqual(layer.weights.data.length, 2, 'Weights should have 2 rows');
            assert.strictEqual(layer.weights.data[0].data.length, 2, 'Each row in weights should have 2 columns');
            assert.strictEqual(layer.bias.data.length, 2, 'Bias should have 2 elements');
        });
    });

    describe('Forward pass', () => {
        it('should correctly compute the forward pass', () => {
            const weights = new Matrix(
                new Vector(0.2, 0.8),
                new Vector(0.5, 0.1)
            );
            const biases = new Vector(0.5, -0.1);

            class MockDenseLayer extends DenseLayer {
                constructor() {
                    super(2, 2); // inputSize and outputSize
                    this.weights = weights;
                    this.bias = biases;
                }
            }

            const layer = new MockDenseLayer();
            const input = new Vector(0.5, 0.9);

            const expectedOutput = new Vector(
                (0.2 * 0.5) + (0.8 * 0.9) + 0.5,  // 0.1 + 0.72 + 0.5 = 1.32
                (0.5 * 0.5) + (0.1 * 0.9) - 0.1   // 0.25 + 0.09 - 0.1 = 0.24
            );

            const output = layer.forward(input);

            assert.deepStrictEqual(output.data, expectedOutput.data, 'Forward pass should compute the correct output');
        });
    });

    describe('Backward pass', () => {
        it('should handle zero gradients correctly', () => {
            const layer = new DenseLayer(2, 2);
            const input = new Vector(0.5, 0.9);
            const output = layer.forward(input);

            const outputGradient = new Vector(0, 0);
            const learningRate = 0.1;

            const inputGradient = layer.backward(outputGradient, learningRate);

            assert.strictEqual(inputGradient.data.length, 2, 'Input gradient should have 2 elements');
        });

        it('should throw error for non-Vector gradient', () => {
            const layer = new DenseLayer(2, 2);
            const input = new Vector(0.5, 0.9);
            layer.forward(input);

            const outputGradient = [0.1, -0.2];
            const learningRate = 0.1;

            assert.throws(() => layer.backward(outputGradient, learningRate), /Argument must be an instance of Vector/, 'Should throw error for non-Vector gradient');
        });

        it('should correctly compute the backward pass', () => {
            const weights = new Matrix(
                new Vector(0.2, 0.8),
                new Vector(0.5, 0.1)
            );
            const biases = new Vector(0.5, -0.1);

            class MockDenseLayer extends DenseLayer {
                constructor() {
                    super(2, 2); // inputSize and outputSize
                    this.weights = weights;
                    this.bias = biases;
                }
            }

            const layer = new MockDenseLayer();
            const input = new Vector(0.5, 0.9);
            const output = layer.forward(input);

            const outputGradient = new Vector(0.1, -0.2);
            const learningRate = 0.1;

            const inputGradient = layer.backward(outputGradient, learningRate);

            // Manually compute the expected input gradient and weights/biases updates
            const expectedInputGradient = new Vector(
                (0.2 * 0.1) + (0.5 * -0.2),
                (0.8 * 0.1) + (0.1 * -0.2)
            );

            const expectedWeights = new Matrix(
                new Vector(0.2 - (0.1 * 0.5 * learningRate), 0.8 - (0.1 * 0.9 * learningRate)),
                new Vector(0.5 - (-0.2 * 0.5 * learningRate), 0.1 - (-0.2 * 0.9 * learningRate))
            );

            const expectedBiases = new Vector(
                0.5 - (0.1 * learningRate),
                -0.1 - (-0.2 * learningRate)
            );

            assert.deepStrictEqual(inputGradient.data, expectedInputGradient.data, 'Backward pass should compute the correct input gradient');
            assert.deepStrictEqual(layer.weights.data, expectedWeights.data, 'Backward pass should update weights correctly');
            assert.deepStrictEqual(layer.bias.data, expectedBiases.data, 'Backward pass should update biases correctly');
        });
    });
});

describe('NN Class', () => {
    describe('Forward pass', () => {
        it('should correctly compute the forward pass', () => {
            const layers = [
                new DenseLayer(2, 2), // inputSize = 2, hiddenSize = 2
                new DenseLayer(2, 1)  // hiddenSize = 2, outputSize = 1
            ];
            const activations = [sigmoidActivation, sigmoidActivation];
            const nn = new NN(layers, activations, mseErrorFunction);
            const input = new Vector(0.5, 0.9);
            const output = nn.forward(input);
            assert.strictEqual(output.data.length, 1, 'Output should be a vector of length 1');
        });
    });

    describe('Training', () => {
        const inputs = new Matrix(
            new Vector(0, 0),
            new Vector(0, 1),
            new Vector(1, 0),
            new Vector(1, 1)
        );
        const outputs = new Matrix(
            new Vector(0),
            new Vector(1),
            new Vector(1),
            new Vector(0)
        );
        const layers = [
            new DenseLayer(2, 2),  // Hidden layer with 2 neurons
            new DenseLayer(2, 1)   // Output layer with 1 neuron
        ];
        const activations = [sigmoidActivation, sigmoidActivation];
        const nn = new NN(layers, activations, mseErrorFunction);
        nn.train(inputs, outputs, 0.1, 10000);
        inputs.data.forEach((inputVector, i) => {
            const output = nn.predict(inputVector);
            const expectedOutput = outputs.data[i].data;
            assert.equal(Math.abs(output.data[0] - expectedOutput[0]) < 0.1, true); // Adjust delta as necessary
        });
    });

    describe('Prediction', () => {
        it('should make predictions', () => {
            const layers = [
                new DenseLayer(2, 2),  // Input layer with 2 neurons
                new DenseLayer(2, 1)   // Output layer with 1 neuron
            ];
            const activations = [sigmoidActivation];
            const nn = new NN(layers, activations, mseErrorFunction);
            const input = new Vector(0.5, 0.9);
            const output = nn.predict(input);
            assert.strictEqual(output.data.length, 1, 'Output should be a vector of length 1');
        });
    });
});
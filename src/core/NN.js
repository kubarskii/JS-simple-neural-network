import fs from 'fs';
import { Vector } from "../utils/math.js";

export default class NN {
    constructor(layers, activations, errorFunction) {
        this.layers = layers;
        this.activations = activations;
        this.errorFunction = errorFunction;
    }

    forward(input) {
        let activations = input;
        this.layerInputs = [input];
        for (let i = 0; i < this.layers.length; i++) {
            const z = this.layers[i].forward(activations);
            this.layerInputs.push(z);
            if (i < this.activations.length) {
                activations = new Vector(...z.data.map(this.activations[i].func));
            } else {
                activations = new Vector(...z.data)
            }
        }
        return activations;
    }

    backward(target, output, learningRate) {
        let outputGradient = new Vector(...this.errorFunction.derivative(target.data, output.data));
        for (let i = this.layers.length - 1; i >= 0; i--) {
            if (i < this.activations.length) {
                const activationDerivatives = new Vector(...this.layerInputs[i + 1].data.map(this.activations[i].derivative));
                outputGradient = new Vector(...outputGradient.data.map((val, idx) => val * activationDerivatives.data[idx]));
            }
            outputGradient = this.layers[i].backward(outputGradient, learningRate);
        }
        return outputGradient;
    }


    train(inputs, targets, learningRate, epochs) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochError = 0;
            inputs.data.forEach((inputVector, i) => {
                const targetVector = targets.data[i];
                const output = this.forward(inputVector);
                const error = this.errorFunction.func(targetVector.data, output.data);
                epochError += error;
                this.backward(targetVector, output, learningRate);
            });

            if (epoch % 1000 === 0) {
                console.log(`Epoch ${epoch}, Error: ${epochError}`);
            }
        }
    }

    predict(input) {
        return this.forward(input);
    }

    saveWeights(filePath) {
        const weights = this.layers.map(layer => ({
            weights: layer.weights.data, // Assuming each layer has a weights attribute
            biases: layer.biases.data     // Assuming each layer has a biases attribute
        }));
        fs.writeFileSync(filePath, JSON.stringify(weights), 'utf8');
        console.log('Weights saved successfully.');
    }

    loadWeights(filePath) {
        const weightsData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        weightsData.forEach((layerData, index) => {
            this.layers[index].setWeights(layerData.weights);
            this.layers[index].setBiases(layerData.biases);
        });
        console.log('Weights loaded successfully.');
    }

}
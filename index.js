import DenseLayer from './denseLayer.js';
import { Vector, Matrix } from './utils/math.js';

const layer = new DenseLayer(3, 2);

const res = layer.forward((new Matrix(new Vector(1, 2, 3))).T);
console.log(layer);
console.log(res);
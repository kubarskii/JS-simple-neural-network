import { Matrix, Vector } from '../utils/math.js';
import { describe, it } from 'node:test';
import assert from 'assert';

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

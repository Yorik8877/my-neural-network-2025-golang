package ai

// Приходит [I_1, I_2, I_3]
// Возвращается [H1_in, H2_in, H3_in]
func (ai *NeuralNetwork) calcNeuronsInput(trainSet []float64) []float64 {
	var result []float64
	for _, weights := range ai.HiddenWeights {
		var halfResult float64
		for i, weight := range weights {
			halfResult += weight * trainSet[i]
		}

		result = append(result, halfResult)
	}

	return result
}

// Приходит [H1_in, H2_in, H3_in]
// Возвращается [H1_out, H2_out, H3_out]
func (ai *NeuralNetwork) calcNeuronsOutput(neuronInputs []float64) []float64 {
	var result []float64
	for _, input := range neuronInputs {
		result = append(result, ai.ActivationFunction(input))
	}

	return result
}

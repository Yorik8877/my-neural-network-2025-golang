package ai

func (ai *NeuralNetwork) TrainSingleEra() {
	for _, trainSet := range ai.dataSet {
		// Считаем входные значения нейронов
		neuronInputs := ai.calcNeuronsInput(trainSet)
		// Считаем выходные значения нейронов
		neuronOutputs := ai.calcNeuronsOutput(neuronInputs)

		// Считаем выходной нейрон
		actualOutput := ai.calcOutput(neuronOutputs)
		// Считаем идеальное значение
		idealOutput := ai.calcIdealFunction(trainSet)

		// Считаем ошибку
		ai.MSE = ai.calcMSE(idealOutput)

		// Начинаем обратный проход
		dO := (idealOutput - actualOutput) * ai.DeactivationFunction(actualOutput)
		dH := make([]float64, len(neuronOutputs))
		for i, neuronOutput := range neuronOutputs {
			dH[i] = dO * ai.OutputWeights[i] * ai.DeactivationFunction(neuronOutput)
		}

		// TODO: implement calc grad
		// TODO: implement calc delta weight
	}
}

// Приходит [H1_out, H2_out, H3_out]
// Возвращается O_out
func (ai *NeuralNetwork) calcOutput(neuronOutputs []float64) float64 {
	var result float64
	for i, weight := range ai.OutputWeights {
		result += weight * neuronOutputs[i]
	}

	return result
}

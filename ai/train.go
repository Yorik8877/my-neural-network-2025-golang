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
		ai.Answers = append(ai.Answers, actualOutput)

		// Считаем ошибку
		ai.MSE = ai.calcMSE(idealOutput)

		// Начинаем обратный проход
		dO := (idealOutput - actualOutput) * ai.DeactivationFunction(actualOutput)
		dHs := make([]float64, len(neuronOutputs))

		// Градиенты выходных синапсов
		outputGradients := make([]float64, len(neuronOutputs))
		for i, neuronOutput := range neuronOutputs {
			dHs[i] = dO * ai.OutputWeights[i] * ai.DeactivationFunction(neuronOutput)
			outputGradients[i] = dO * neuronOutput
		}

		// Градиенты скрытых синапсов
		var hiddenGradients [][]float64
		for _, dH := range dHs {
			var buffer []float64
			for _, inputValue := range trainSet {
				buffer = append(buffer, dH*inputValue)
			}
			hiddenGradients = append(hiddenGradients, buffer)
		}

		// Неявный подсчёт дельт весов
		ai.calcGradients(hiddenGradients, outputGradients)

		// Подсчитать новые выходные веса
		for idx, outputWeight := range ai.OutputWeights {
			ai.OutputWeights[idx] = outputWeight + ai.PreviousOutputWeightDeltas[idx]
		}

		// Подсчитать новые скрытые веса
		for i, weights := range ai.HiddenWeights {
			for j, weight := range weights {
				ai.HiddenWeights[i][j] = weight + ai.PreviousHiddenWeightDeltas[i][j]
			}
		}

		break
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

func (ai NeuralNetwork) calcDelta(grad, previousDelta float64) float64 {
	return ai.E*grad + ai.A*previousDelta
}

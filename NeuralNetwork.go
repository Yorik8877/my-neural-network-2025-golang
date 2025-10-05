package main

type NeuralNetwork struct {
	// Транспонированная матрица весов скрытых синапсов
	HiddenWeights [][]float64
	// Вектор весов выходных синапсов
	OutputWeights []float64

	// Функции активации
	ActivationFunction func(float64) float64
	// Конечно-разностный аналог функции активации для обратного прохода
	DeactivationFunction func(float64) float64
	calcIdealFunction    func([]float64) float64

	RequestedEra uint

	// Среднеквадратичная ошибка
	MSE float64
	// Коэффициент обучения
	E float64
	// Момент обучения
	a float64

	dataSet [][]float64
	answers []float64

	// Дельты скрытых синапсов
	previousHiddenWeightDeltas [][]float64
	// Дельты выходных синапсов
	previousOutputWeightDeltas []float64
}

func NewNeuralNetwork(
	hiddenWeights [][]float64, outputWeights []float64,
	activationFunction, deactivationFunction func(float64) float64,
	calcIdealFunction func([]float64) float64,
	requestedEra uint, E, a float64,
) *NeuralNetwork {
	return &NeuralNetwork{
		HiddenWeights:        hiddenWeights,
		OutputWeights:        outputWeights,
		ActivationFunction:   activationFunction,
		DeactivationFunction: deactivationFunction,
		calcIdealFunction:    calcIdealFunction,
		RequestedEra:         requestedEra,
	}
}

func (ai *NeuralNetwork) ReadFromCSV(filePath string) error {
	return nil
}

func (ai *NeuralNetwork) TrainAI() error {
	for _, trainSet := range ai.dataSet {
		// Получение идеального значения
		idealValue := ai.calcIdealFunction(trainSet)

		// Подсчёт значений на входе скрытых синапсов
		hiddenInputsValues := ai.calcHiddenInputsValues(trainSet)
		// Подсчёт значений на выходе скрытых синапсов (Нормализация)
		hiddenOutputsValues := ai.calcHiddenOutputsValues(hiddenInputsValues)

		// Подсчёт значений на выходе
		outputValue := ai.calcOutput(hiddenOutputsValues)

		// Запись ответа для подсчёта MSE
		ai.answers = append(ai.answers, outputValue)
		// Подсчёт MSE
		ai.MSE = ai.calcMSE(idealValue)

		// Рассчёт градиентов
		hiddenGradients, outputGradients := ai.calcGradients(trainSet, idealValue, outputValue)
		// Рассчёт дельт
		hiddenDeltas, outputDeltas := ai.calcDeltas(hiddenGradients, outputGradients)
		ai.previousHiddenWeightDeltas = hiddenDeltas
		ai.previousOutputWeightDeltas = outputDeltas

		// Рассчёт новых весов
		newHiddenWeight, newOutputWeights := ai.calcNewWeights(hiddenDeltas, outputDeltas)
		ai.HiddenWeights = newHiddenWeight
		ai.OutputWeights = newOutputWeights
	}

	return nil
}

func (ai *NeuralNetwork) calcHiddenInputsValues(trainSet []float64) []float64 {
	var result []float64
	for _, weights := range ai.HiddenWeights {
		halfResult := float64(0)
		for i := range weights {
			halfResult += weights[i] * trainSet[i]
		}

		result = append(result, halfResult)
	}

	return result
}

func (ai *NeuralNetwork) calcHiddenOutputsValues(inputs []float64) []float64 {
	var result []float64
	for _, inputValue := range inputs {
		result = append(result, ai.ActivationFunction(inputValue))
	}

	return result
}

func (ai *NeuralNetwork) calcOutput(hiddenOutputs []float64) float64 {
	var result float64
	for idx := range hiddenOutputs {
		result += hiddenOutputs[idx] * ai.OutputWeights[idx]
	}

	return result
}

func (ai *NeuralNetwork) calcGradients(trainSet []float64, ideal, real float64) ([][]float64, []float64) {
	dO := (ideal - real) * ai.DeactivationFunction(real)

	var dHs []float64
	for idx := range ai.OutputWeights {
		dHs = append(dHs, dO*ai.OutputWeights[idx])
	}

	var hiddenGradients [][]float64
	var outputGradients []float64
	for _, dH := range dHs {
		var buffer []float64
		for _, inputValue := range trainSet {
			buffer = append(buffer, inputValue*dH)
		}

		hiddenGradients = append(hiddenGradients, buffer)
		outputGradients = append(outputGradients, dH*dO)
	}

	return hiddenGradients, outputGradients
}

func (ai *NeuralNetwork) calcDeltas(hiddenGradients [][]float64, outputGradients []float64) ([][]float64, []float64) {
	var hiddenDeltas [][]float64

	for lineOfGradientsIdx, lineOfGradients := range hiddenGradients {
		var buffer []float64
		for idx, gradient := range lineOfGradients {
			var previousHiddenDelta = float64(0)
			if ai.previousHiddenWeightDeltas != nil && ai.previousHiddenWeightDeltas[lineOfGradientsIdx] != nil {
				previousHiddenDelta = ai.previousHiddenWeightDeltas[lineOfGradientsIdx][idx]

			}

			delta := ai.E*gradient + ai.a*previousHiddenDelta
			buffer = append(buffer, delta)
		}

		hiddenDeltas = append(hiddenDeltas, buffer)
	}

	var outputDeltas []float64
	for idx, gradient := range outputGradients {
		var previousOutputDelta = float64(0)
		if ai.previousOutputWeightDeltas != nil {
			previousOutputDelta = ai.previousOutputWeightDeltas[idx]
		}

		delta := ai.E*gradient + ai.a*previousOutputDelta
		outputDeltas = append(outputDeltas, delta)
	}

	return hiddenDeltas, outputDeltas
}

func (ai *NeuralNetwork) calcNewWeights(hiddenDeltas [][]float64, outputDeltas []float64) ([][]float64, []float64) {
	var newHiddenWeights [][]float64
	for lineOfDeltasIdx, lineOfDeltas := range hiddenDeltas {
		var buffer []float64
		for idx, delta := range lineOfDeltas {
			buffer = append(buffer, ai.HiddenWeights[lineOfDeltasIdx][idx]+delta)
		}

		newHiddenWeights = append(newHiddenWeights, buffer)
	}

	var newOutputWeights []float64
	for idx, delta := range outputDeltas {
		newOutputWeights = append(newOutputWeights, ai.OutputWeights[idx]+delta)
	}

	return newHiddenWeights, newOutputWeights
}

func (ai *NeuralNetwork) calcMSE(ideal float64) float64 {
	mse := float64(0)
	for _, answer := range ai.answers {
		mse += (ideal - answer) * (ideal - answer)
	}

	return mse / float64(len(ai.answers))
}

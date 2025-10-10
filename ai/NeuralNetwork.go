package ai

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
	A float64

	dataSet [][]float64
	Answers []float64

	// Дельты скрытых синапсов
	PreviousHiddenWeightDeltas [][]float64
	// Дельты выходных синапсов
	PreviousOutputWeightDeltas []float64
}

func NewNeuralNetwork(
	hiddenWeights [][]float64, outputWeights []float64,
	activationFunction, deactivationFunction func(float64) float64,
	calcIdealFunction func([]float64) float64,
) *NeuralNetwork {
	return &NeuralNetwork{
		HiddenWeights:        hiddenWeights,
		OutputWeights:        outputWeights,
		ActivationFunction:   activationFunction,
		DeactivationFunction: deactivationFunction,
		calcIdealFunction:    calcIdealFunction,
		E:                    0.7,
		A:                    0.3,
	}
}

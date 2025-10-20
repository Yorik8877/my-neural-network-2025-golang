package ai

type NeuralNetwork struct {
	// Транспонированная матрица весов скрытых синапсов
	HiddenWeights [][]float64 `json:"hidden_weights"`
	// Вектор весов выходных синапсов
	OutputWeights []float64 `json:"output_weights"`

	// Функции активации
	ActivationFunction func(float64) (float64, error) `json:"-"`
	// Конечно-разностный аналог функции активации для обратного прохода
	DeactivationFunction func(float64) (float64, error)   `json:"-"`
	calcIdealFunction    func([]float64) (float64, error) `json:"-"`

	RequestedEra uint `json:"requested_era"`

	// Среднеквадратичная ошибка
	MSE float64 `json:"mse"`
	// Коэффициент обучения
	E float64 `json:"E"`
	// Момент обучения
	A float64 `json:"a"`

	dataSet [][]float64
	Answers []float64 `json:"answers"`

	// Дельты скрытых синапсов
	PreviousHiddenWeightDeltas [][]float64 `json:"previous_hidden_weights_deltas"`
	// Дельты выходных синапсов
	PreviousOutputWeightDeltas []float64 `json:"previous_output_weights_deltas"`
}

func NewNeuralNetwork(
	hiddenWeights [][]float64, outputWeights []float64,
	activationFunction, deactivationFunction func(float64) (float64, error),
	calcIdealFunction func([]float64) (float64, error),
) *NeuralNetwork {
	baseLength := len(hiddenWeights)
	for _, weights := range hiddenWeights {
		if baseLength != len(weights) {
			panic("Неверно заданы веса. Требуется квадратная матрица!")
		}
	}

	newPreviousHiddenWeightDeltas := make([][]float64, baseLength)
	for idx := range newPreviousHiddenWeightDeltas {
		newPreviousHiddenWeightDeltas[idx] = make([]float64, baseLength)
	}
	newPreviousOutputWeightDeltas := make([]float64, len(outputWeights))

	return &NeuralNetwork{
		HiddenWeights:              hiddenWeights,
		OutputWeights:              outputWeights,
		ActivationFunction:         activationFunction,
		DeactivationFunction:       deactivationFunction,
		calcIdealFunction:          calcIdealFunction,
		E:                          0.7,
		A:                          0.3,
		PreviousHiddenWeightDeltas: newPreviousHiddenWeightDeltas,
		PreviousOutputWeightDeltas: newPreviousOutputWeightDeltas,
	}
}

// Shared training status module (per model)
type TrainingState = {
  isTraining: boolean;
  progress: number;
  message: string;
  error: string | null;
};

const defaultState: TrainingState = {
  isTraining: false,
  progress: 0,
  message: "",
  error: null,
};

const trainingStatus: Record<string, TrainingState> = {};

function getState(model: string): TrainingState {
  return trainingStatus[model] ?? { ...defaultState };
}

export function setTrainingStatus(model: string, status: Partial<TrainingState>) {
  trainingStatus[model] = { ...getState(model), ...status };
}

export function getTrainingStatus(model: string) {
  return getState(model);
}

export function getAllTrainingStatus() {
  return trainingStatus;
}


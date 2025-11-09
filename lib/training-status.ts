// Shared training status module

let trainingStatus = {
  isTraining: false,
  progress: 0,
  message: "",
  error: null as string | null,
};

export function setTrainingStatus(status: Partial<typeof trainingStatus>) {
  trainingStatus = { ...trainingStatus, ...status };
}

export function getTrainingStatus() {
  return trainingStatus;
}


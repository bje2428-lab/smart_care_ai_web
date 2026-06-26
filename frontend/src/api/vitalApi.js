export const VITAL_API_URL =
  import.meta.env.VITE_API_URL || import.meta.env.VITE_BACKEND_URL;

function getApiUrl() {
  if (!VITAL_API_URL) {
    throw new Error(
      "VITE_API_URL이 설정되지 않았습니다. frontend/.env 파일에 실제 백엔드 주소를 넣어주세요.",
    );
  }

  return VITAL_API_URL.replace(/\/$/, "");
}

async function request(path, options = {}) {
  const baseUrl = getApiUrl();
  const url = `${baseUrl}${path}`;

  const isFormData = options.body instanceof FormData;

  const response = await fetch(url, {
    ...options,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });

  let data = null;

  try {
    data = await response.json();
  } catch (error) {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.detail || data?.message || `API 요청 실패: ${response.status}`;

    throw new Error(message);
  }

  return data;
}

export function getVitalHealth() {
  return request("/vital/health");
}

export function getVitalFeatures() {
  return request("/vital/features");
}

export function predictVital(features) {
  return request("/vital/predict", {
    method: "POST",
    body: JSON.stringify({ features }),
  });
}

export function predictVitalCsv(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/vital/predict-file", {
    method: "POST",
    body: formData,
  });
}

export function uploadVitalSimulationCsv(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/vital/simulation/upload", {
    method: "POST",
    body: formData,
  });
}

export function getVitalSimulationStatus() {
  return request("/vital/simulation/status");
}

export function resetVitalSimulation() {
  return request("/vital/simulation/reset", {
    method: "POST",
  });
}

export function clearVitalSimulation() {
  return request("/vital/simulation/clear", {
    method: "POST",
  });
}

export function nextVitalSimulation() {
  return request("/vital/simulation/next", {
    method: "POST",
  });
}

export function getVitalHistory(limit = 30) {
  return request(`/vital/history?limit=${limit}`);
}

export function getVitalStats() {
  return request("/vital/stats");
}

export function resetVitalCounter() {
  return request("/vital/reset", {
    method: "POST",
  });
}

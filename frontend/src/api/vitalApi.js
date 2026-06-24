const rawApiUrls =
  import.meta.env.VITE_API_URLS ||
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_BACKEND_URL ||
  "";

const API_CANDIDATES = rawApiUrls
  .split(",")
  .map((url) => url.trim())
  .filter(Boolean)
  .map((url) => url.replace(/\/$/, ""));

let selectedApiUrl = null;

function getErrorMessage(data, status) {
  const message =
    data?.detail?.message ||
    data?.detail ||
    data?.message ||
    `API 요청 실패: ${status}`;

  return typeof message === "string" ? message : JSON.stringify(message);
}

async function resolveApiUrl() {
  if (selectedApiUrl) {
    return selectedApiUrl;
  }

  if (API_CANDIDATES.length === 0) {
    throw new Error(
      ".env에 VITE_API_URLS 또는 VITE_API_URL이 설정되어 있지 않습니다.",
    );
  }

  for (const baseUrl of API_CANDIDATES) {
    try {
      const response = await fetch(`${baseUrl}/vital/health`);

      if (response.ok) {
        selectedApiUrl = baseUrl;
        return selectedApiUrl;
      }
    } catch (error) {
      // 다음 후보 주소 확인
    }
  }

  throw new Error(
    `사용 가능한 Vital API 서버를 찾지 못했습니다. 확인한 주소: ${API_CANDIDATES.join(
      ", ",
    )}`,
  );
}

async function request(path, options = {}) {
  const baseUrl = await resolveApiUrl();
  const url = `${baseUrl}${path}`;

  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let data = null;

  try {
    data = await response.json();
  } catch (error) {
    data = null;
  }

  if (!response.ok) {
    throw new Error(getErrorMessage(data, response.status));
  }

  return data;
}

export function getSelectedVitalApiUrl() {
  return selectedApiUrl;
}

export function getVitalApiCandidates() {
  return API_CANDIDATES;
}

export async function getVitalHealth() {
  return request("/vital/health");
}

export async function getVitalFeatures() {
  return request("/vital/features");
}

export async function predictVital(features) {
  return request("/vital/predict", {
    method: "POST",
    body: JSON.stringify({
      features: features.map((value) => Number(value)),
    }),
  });
}

export async function getVitalHistory(limit = 20) {
  return request(`/vital/history?limit=${limit}`);
}

export async function getVitalStats() {
  return request("/vital/stats");
}

export async function reloadVitalModel() {
  return request("/vital/reload", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

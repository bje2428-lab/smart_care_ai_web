const API_URLS = (
  import.meta.env.VITE_API_URLS ||
  import.meta.env.VITE_API_URL ||
  ""
)
  .split(",")
  .map((url) => url.trim())
  .filter(Boolean);

let activeApiUrl = null;

async function checkUrlHealth(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 2000);

  try {
    const response = await fetch(`${url}/health`, {
      method: "GET",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    clearTimeout(timeoutId);
    return false;
  }
}

export async function getActiveApiUrl() {
  if (activeApiUrl) {
    return activeApiUrl;
  }

  if (API_URLS.length === 0) {
    throw new Error(
      ".env에 VITE_API_URL 또는 VITE_API_URLS가 설정되어 있지 않습니다.",
    );
  }

  for (const url of API_URLS) {
    const isHealthy = await checkUrlHealth(url);

    if (isHealthy) {
      activeApiUrl = url;
      console.log("연결된 백엔드:", activeApiUrl);
      return activeApiUrl;
    }
  }

  throw new Error("사용 가능한 백엔드 서버가 없습니다.");
}

export function resetActiveApiUrl() {
  activeApiUrl = null;
}

export async function predictFall(csvFile) {
  const apiUrl = await getActiveApiUrl();

  const formData = new FormData();
  formData.append("file", csvFile);

  const response = await fetch(`${apiUrl}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "예측 요청에 실패했습니다.";

    try {
      const errorData = await response.json();
      message = errorData.detail || message;
    } catch {
      // JSON 에러가 아니면 기본 메시지 사용
    }

    throw new Error(message);
  }

  return response.json();
}

export async function checkBackendHealth() {
  const apiUrl = await getActiveApiUrl();

  const response = await fetch(`${apiUrl}/health`);

  if (!response.ok) {
    throw new Error("백엔드 서버에 연결할 수 없습니다.");
  }

  return response.json();
}

export async function getStats() {
  const apiUrl = await getActiveApiUrl();

  const response = await fetch(`${apiUrl}/stats`);

  if (!response.ok) {
    throw new Error("통계 조회에 실패했습니다.");
  }

  return response.json();
}

export async function getEvents(limit = 100) {
  const apiUrl = await getActiveApiUrl();

  const response = await fetch(`${apiUrl}/events?limit=${limit}`);

  if (!response.ok) {
    throw new Error("이벤트 로그 조회에 실패했습니다.");
  }

  return response.json();
}

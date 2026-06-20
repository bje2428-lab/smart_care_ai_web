const getRequiredEnv = (key) => {
    const value = import.meta.env[key];
  
    if (!value) {
      throw new Error(
        `${key} 환경변수가 설정되지 않았습니다. frontend/.env 파일을 확인하세요.`
      );
    }
  
    return value.replace(/\/+$/, "");
  };
  
  export const API_URL = getRequiredEnv("VITE_API_URL");
  export const ABNORMAL_API_URL = API_URL;
  
  async function requestJson(path, options = {}) {
    const response = await fetch(`${API_URL}${path}`, options);
  
    let data = null;
  
    try {
      data = await response.json();
    } catch {
      data = null;
    }
  
    if (!response.ok) {
      throw new Error(
        data?.message ||
          data?.detail ||
          `API 요청 실패: ${response.status} ${path}`
      );
    }
  
    return data;
  }
  
  export async function getAbnormalHealth() {
    return requestJson("/abnormal/health");
  }
  
  export async function getAbnormalFeatures() {
    return requestJson("/abnormal/features");
  }
  
  export async function getAbnormalHistory(limit = 20) {
    return requestJson(`/abnormal/history?limit=${limit}`);
  }
  
  export async function getAbnormalStats() {
    return requestJson("/abnormal/stats");
  }
  
  export async function deleteAbnormalHistory() {
    return requestJson("/abnormal/history", {
      method: "DELETE",
    });
  }
  
  export async function uploadAbnormalSimulationFile(file) {
    const formData = new FormData();
    formData.append("file", file);
  
    return requestJson("/abnormal/simulation/upload", {
      method: "POST",
      body: formData,
    });
  }
  
  export async function getAbnormalSimulationStatus() {
    return requestJson("/abnormal/simulation/status");
  }
  
  export async function resetAbnormalSimulation() {
    return requestJson("/abnormal/simulation/reset", {
      method: "POST",
    });
  }
  
  export async function runNextAbnormalSimulation() {
    return requestJson("/abnormal/simulation/next", {
      method: "POST",
    });
  }
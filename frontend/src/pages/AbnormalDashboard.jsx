import { useEffect, useMemo, useRef, useState } from "react";
import {
  deleteAbnormalHistory,
  getAbnormalFeatures,
  getAbnormalHealth,
  getAbnormalHistory,
  getAbnormalSimulationStatus,
  getAbnormalStats,
  resetAbnormalSimulation,
  runNextAbnormalSimulation,
  uploadAbnormalSimulationFile,
} from "../api/abnormalApi";
import "./AbnormalDashboard.css";

const HISTORY_PAGE_SIZE = 5;
const AUTO_INTERVAL_MS = 10000;

const SENSOR_GROUPS = [
  {
    key: "environment",
    title: "실내 환경 정보",
    icon: "🏠",
    desc: "온도, 습도, 조도, 공기질",
    metrics: [
      {
        key: "Temperature",
        label: "실내 온도",
        icon: "🌡️",
        unit: "℃",
        decimals: 1,
      },
      {
        key: "Humidity",
        label: "습도",
        icon: "💧",
        unit: "%",
        decimals: 0,
      },
      {
        key: "Illuminance",
        label: "조도",
        icon: "💡",
        unit: "lux",
        decimals: 0,
      },
      {
        key: "CO2",
        label: "이산화탄소",
        icon: "🌫️",
        unit: "ppm",
        decimals: 0,
      },
      {
        key: "TVOC",
        label: "공기질",
        icon: "🍃",
        unit: "",
        decimals: 0,
      },
    ],
  },
  {
    key: "vital",
    title: "생체 신호 정보",
    icon: "❤️",
    desc: "심박, 호흡, 산소포화도, 피부 온도",
    metrics: [
      {
        key: "HeartRate",
        label: "심박수",
        icon: "💓",
        unit: "bpm",
        decimals: 0,
      },
      {
        key: "BreathRate",
        label: "호흡수",
        icon: "🫁",
        unit: "회/분",
        decimals: 0,
      },
      {
        key: "SPO2",
        label: "산소포화도",
        icon: "🩸",
        unit: "%",
        decimals: 0,
      },
      {
        key: "SkinTemperature",
        label: "피부 온도",
        icon: "🤒",
        unit: "℃",
        decimals: 1,
      },
      {
        key: "StressIndex",
        label: "스트레스 지수",
        icon: "⚡",
        unit: "점",
        decimals: 0,
      },
    ],
  },
  {
    key: "activity",
    title: "활동 정보",
    icon: "🚶",
    desc: "움직임, 걸음 수, 활동 강도",
    metrics: [
      {
        key: "Activity_IR",
        label: "움직임 감지",
        icon: "📡",
        type: "binary",
        onText: "감지",
        offText: "미감지",
      },
      {
        key: "WalkingSteps",
        label: "걸음 수",
        icon: "👣",
        unit: "걸음",
        decimals: 0,
      },
      {
        key: "ActivityIntensity",
        label: "활동 강도",
        icon: "🏃",
        unit: "점",
        decimals: 0,
      },
      {
        key: "CaloricExpenditure",
        label: "칼로리 소모",
        icon: "🔥",
        unit: "kcal",
        decimals: 1,
      },
      {
        key: "Button",
        label: "비상 버튼",
        icon: "🆘",
        type: "binary",
        onText: "눌림",
        offText: "없음",
      },
      {
        key: "Shout",
        label: "외침 감지",
        icon: "📢",
        type: "binary",
        onText: "감지",
        offText: "없음",
      },
    ],
  },
];

function getStateClass(state) {
  if (state === "위험") return "danger";
  if (state === "주의") return "warning";
  if (state === "외출") return "outing";
  if (state === "식사") return "meal";
  if (state === "수면") return "sleep";
  return "normal";
}

function getRiskText(score) {
  const value = Number(score || 0);

  if (value >= 80) return "위험";
  if (value >= 50) return "주의";
  return "정상";
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "-";

  const num = Number(value);

  if (Number.isNaN(num)) return String(value);

  return Number.isInteger(num) ? String(num) : num.toFixed(2);
}

function hasSensorValue(sensor, key) {
  return (
    sensor &&
    sensor[key] !== undefined &&
    sensor[key] !== null &&
    sensor[key] !== ""
  );
}

function getSensorNumber(sensor, key) {
  if (!hasSensorValue(sensor, key)) return null;

  const num = Number(sensor[key]);

  if (Number.isNaN(num)) return null;

  return num;
}

function formatSensorValue(sensor, metric) {
  if (!hasSensorValue(sensor, metric.key)) {
    return "-";
  }

  const rawValue = sensor[metric.key];

  if (metric.type === "binary") {
    const num = Number(rawValue);
    return num >= 1 ? metric.onText || "있음" : metric.offText || "없음";
  }

  const num = Number(rawValue);

  if (Number.isNaN(num)) {
    return String(rawValue);
  }

  const decimals = metric.decimals ?? 0;

  const valueText =
    Number.isInteger(num) && decimals === 0
      ? String(num)
      : num.toFixed(decimals);

  return metric.unit ? `${valueText} ${metric.unit}` : valueText;
}

function getSensorStatus(sensor, metric) {
  const value = getSensorNumber(sensor, metric.key);

  if (value === null) return "neutral";

  switch (metric.key) {
    case "Temperature":
      if (value < 15 || value > 32) return "danger";
      if (value < 18 || value > 29) return "warning";
      return "good";

    case "Humidity":
      if (value < 20 || value > 90) return "danger";
      if (value < 30 || value > 80) return "warning";
      return "good";

    case "Illuminance":
      if (value < 30) return "warning";
      return "good";

    case "CO2":
      if (value >= 1500) return "danger";
      if (value >= 1000) return "warning";
      return "good";

    case "TVOC":
      if (value >= 1000) return "danger";
      if (value >= 500) return "warning";
      return "good";

    case "HeartRate":
      if (value < 45 || value > 120) return "danger";
      if (value < 55 || value > 100) return "warning";
      return "good";

    case "BreathRate":
      if (value < 8 || value > 25) return "danger";
      if (value < 12 || value > 22) return "warning";
      return "good";

    case "SPO2":
      if (value < 90) return "danger";
      if (value < 95) return "warning";
      return "good";

    case "SkinTemperature":
      if (value < 34 || value > 38) return "danger";
      if (value < 35 || value > 37.5) return "warning";
      return "good";

    case "StressIndex":
      if (value >= 80) return "danger";
      if (value >= 60) return "warning";
      return "good";

    case "Activity_IR":
      return value >= 1 ? "good" : "neutral";

    case "Button":
    case "Shout":
      return value >= 1 ? "danger" : "good";

    case "ActivityIntensity":
      if (value >= 8) return "warning";
      return "good";

    default:
      return "neutral";
  }
}

function getStatusText(status) {
  if (status === "danger") return "위험";
  if (status === "warning") return "주의";
  if (status === "good") return "정상";
  return "확인";
}

function SensorCompactRow({ sensor, metric }) {
  const status = getSensorStatus(sensor, metric);
  const valueText = formatSensorValue(sensor, metric);

  return (
    <div className={`sensor-compact-row ${status}`}>
      <div className="sensor-compact-left">
        <span className="sensor-compact-icon">{metric.icon}</span>
        <span className="sensor-compact-label">{metric.label}</span>
      </div>

      <strong>{valueText}</strong>

      <em className={`sensor-compact-status ${status}`}>
        {getStatusText(status)}
      </em>
    </div>
  );
}

function SensorInfoSection({ sensor }) {
  if (!sensor) {
    return (
      <section className="abnormal-sensor-dashboard">
        <div className="abnormal-sensor-empty">
          아직 표시할 센서 데이터가 없습니다.
          <br />
          CSV 파일을 업로드하고 예측을 실행하면 실내 환경, 생체 신호, 활동
          정보가 표시됩니다.
        </div>
      </section>
    );
  }

  return (
    <section className="abnormal-sensor-dashboard">
      <div className="abnormal-sensor-title-row">
        <div>
          <h2>센서 상세 정보</h2>
          <p>현재 예측에 사용된 센서값을 3개 영역으로 압축해서 보여줍니다.</p>
        </div>
      </div>

      <div className="sensor-compact-grid">
        {SENSOR_GROUPS.map((group) => (
          <article className="sensor-compact-card" key={group.key}>
            <div className="sensor-compact-header">
              <div className="sensor-compact-header-icon">{group.icon}</div>

              <div>
                <h3>{group.title}</h3>
                <p>{group.desc}</p>
              </div>
            </div>

            <div className="sensor-compact-list">
              {group.metrics.map((metric) => (
                <SensorCompactRow
                  key={`${group.key}-${metric.key}`}
                  sensor={sensor}
                  metric={metric}
                />
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function AbnormalDashboard() {
  const [health, setHealth] = useState(null);
  const [features, setFeatures] = useState([]);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [simulationStatus, setSimulationStatus] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadInfo, setUploadInfo] = useState(null);

  const [latestResult, setLatestResult] = useState(null);
  const [selectedHistoryResult, setSelectedHistoryResult] = useState(null);

  const [pageLoading, setPageLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [autoRunning, setAutoRunning] = useState(false);
  const [error, setError] = useState("");

  const [historyPage, setHistoryPage] = useState(1);

  const autoTimerRef = useRef(null);
  const autoLockRef = useRef(false);

  const displayResult = selectedHistoryResult || latestResult;

  const totalHistoryPages = useMemo(() => {
    return Math.max(1, Math.ceil(history.length / HISTORY_PAGE_SIZE));
  }, [history]);

  const currentHistoryPage = Math.min(historyPage, totalHistoryPages);

  const pagedHistory = useMemo(() => {
    const start = (currentHistoryPage - 1) * HISTORY_PAGE_SIZE;
    const end = start + HISTORY_PAGE_SIZE;
    return history.slice(start, end);
  }, [history, currentHistoryPage]);

  useEffect(() => {
    if (historyPage > totalHistoryPages) {
      setHistoryPage(totalHistoryPages);
    }
  }, [historyPage, totalHistoryPages]);

  const stopAutoSimulation = () => {
    if (autoTimerRef.current) {
      clearInterval(autoTimerRef.current);
      autoTimerRef.current = null;
    }

    autoLockRef.current = false;
    setAutoRunning(false);
  };

  useEffect(() => {
    return () => {
      if (autoTimerRef.current) {
        clearInterval(autoTimerRef.current);
        autoTimerRef.current = null;
      }

      autoLockRef.current = false;
    };
  }, []);

  const loadPageData = async () => {
    try {
      setError("");
      setPageLoading(true);

      const [healthData, featureData, historyData, statsData, simData] =
        await Promise.all([
          getAbnormalHealth(),
          getAbnormalFeatures(),
          getAbnormalHistory(100),
          getAbnormalStats(),
          getAbnormalSimulationStatus(),
        ]);

      setHealth(healthData);
      setFeatures(featureData.features || []);
      setHistory(historyData.items || []);
      setStats(statsData);
      setSimulationStatus(simData);
      setHistoryPage(1);
    } catch (err) {
      setError(
        "이상행동 API 연결에 실패했습니다. 8000번 백엔드 서버와 /abnormal API 연결을 확인하세요.",
      );
      console.error(err);
    } finally {
      setPageLoading(false);
    }
  };

  useEffect(() => {
    loadPageData();
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setSelectedFile(file || null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("업로드할 CSV 또는 Excel 파일을 선택하세요.");
      return;
    }

    try {
      stopAutoSimulation();
      setError("");
      setUploading(true);
      setSelectedHistoryResult(null);
      setLatestResult(null);

      const data = await uploadAbnormalSimulationFile(selectedFile);
      setUploadInfo(data);

      const simData = await getAbnormalSimulationStatus();
      setSimulationStatus(simData);
      setHistoryPage(1);
    } catch (err) {
      setError(
        String(err.message || "시뮬레이션 파일 업로드 중 오류가 발생했습니다."),
      );
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const refreshAfterPrediction = async (newLatestResult = null) => {
    const [historyData, statsData, simData] = await Promise.all([
      getAbnormalHistory(100),
      getAbnormalStats(),
      getAbnormalSimulationStatus(),
    ]);

    setHistory(historyData.items || []);
    setStats(statsData);
    setSimulationStatus(simData);
    setHistoryPage(1);

    if (newLatestResult) {
      setLatestResult(newLatestResult);
      setSelectedHistoryResult(null);
    }

    return simData;
  };

  const runOnePrediction = async ({ isAuto = false } = {}) => {
    if (autoLockRef.current) return null;

    try {
      autoLockRef.current = true;
      setError("");
      setPredicting(true);

      const data = await runNextAbnormalSimulation();
      const simData = await refreshAfterPrediction(data);

      if (isAuto && Number(simData?.remaining || 0) <= 0) {
        stopAutoSimulation();
        setError(
          "모든 데이터를 확인했습니다. 다시 보려면 '처음부터'를 눌러주세요.",
        );
      }

      return simData;
    } catch (err) {
      const message = String(
        err.message || "다음 센서 데이터 예측 중 오류가 발생했습니다.",
      );

      setError(message);
      console.error(err);

      if (isAuto) {
        stopAutoSimulation();
      }

      return null;
    } finally {
      autoLockRef.current = false;
      setPredicting(false);
    }
  };

  const handleNext = async () => {
    await runOnePrediction({ isAuto: false });
  };

  const handleAutoToggle = async () => {
    if (autoRunning) {
      stopAutoSimulation();
      return;
    }

    try {
      setError("");
      setSelectedHistoryResult(null);

      const simData = await getAbnormalSimulationStatus();

      if (!simData?.loaded) {
        setError("먼저 CSV 또는 Excel 파일을 업로드하세요.");
        return;
      }

      const remaining = Number(simData.remaining || 0);

      if (remaining <= 0) {
        setError(
          "남은 데이터가 없습니다. '처음부터'를 누른 뒤 다시 실행하세요.",
        );
        return;
      }

      setAutoRunning(true);

      const latestSimData = await runOnePrediction({ isAuto: true });

      if (latestSimData && Number(latestSimData.remaining || 0) > 0) {
        autoTimerRef.current = setInterval(() => {
          runOnePrediction({ isAuto: true });
        }, AUTO_INTERVAL_MS);
      } else {
        stopAutoSimulation();
      }
    } catch (err) {
      setError(
        String(err.message || "10초 자동 예측 실행 중 오류가 발생했습니다."),
      );
      console.error(err);
      stopAutoSimulation();
    }
  };

  const handleResetSimulation = async () => {
    try {
      stopAutoSimulation();
      setError("");
      await resetAbnormalSimulation();

      const simData = await getAbnormalSimulationStatus();
      setSimulationStatus(simData);
      setLatestResult(null);
      setSelectedHistoryResult(null);
    } catch (err) {
      setError(
        String(err.message || "시뮬레이션 초기화 중 오류가 발생했습니다."),
      );
      console.error(err);
    }
  };

  const handleDeleteHistory = async () => {
    const ok = window.confirm("위험/주의 이상행동 기록을 모두 삭제할까요?");

    if (!ok) return;

    try {
      await deleteAbnormalHistory();

      setHistory([]);
      setLatestResult(null);
      setSelectedHistoryResult(null);
      setHistoryPage(1);
      setStats({
        total: 0,
        danger_count: 0,
        warning_count: 0,
        outing_count: 0,
        meal_count: 0,
        sleep_count: 0,
        guardian_alert_count: 0,
      });
    } catch (err) {
      setError("기록 삭제 중 오류가 발생했습니다.");
      console.error(err);
    }
  };

  const handlePrevHistoryPage = () => {
    setHistoryPage((prev) => Math.max(1, prev - 1));
  };

  const handleNextHistoryPage = () => {
    setHistoryPage((prev) => Math.min(totalHistoryPages, prev + 1));
  };

  const progressText = useMemo(() => {
    if (!simulationStatus?.loaded) return "-";
    return `${simulationStatus.current_index} / ${simulationStatus.rows}`;
  }, [simulationStatus]);

  const canRunNext =
    simulationStatus?.loaded &&
    Number(simulationStatus.remaining || 0) > 0 &&
    !predicting &&
    !autoRunning;

  const canRunAuto =
    simulationStatus?.loaded &&
    Number(simulationStatus.remaining || 0) > 0 &&
    !predicting &&
    !uploading;

  return (
    <main className="abnormal-page">
      <section className="abnormal-hero">
        <div>
          <h1>이상행동 관제 대시보드</h1>
          <p className="abnormal-hero-desc">
            CSV 또는 Excel 센서 데이터를 업로드하고 10초 간격으로 예측하여 위험,
            주의, 외출, 식사, 수면 상태를 분석합니다.
          </p>
        </div>
      </section>

      {error && <div className="abnormal-alert error">{error}</div>}

      {pageLoading ? (
        <div className="abnormal-loading">
          이상행동 모델 정보를 불러오는 중...
        </div>
      ) : (
        <>
          <section className="abnormal-summary-grid">
            <div className="abnormal-summary-card">
              <p>API 상태</p>
              <strong>
                {health?.status === "ok" ? "정상 연결" : "확인 필요"}
              </strong>
              <span>{health?.message || "이상행동 API 상태 확인"}</span>
            </div>

            <div className="abnormal-summary-card">
              <p>필요 Feature</p>
              <strong>{features.length}</strong>
              <span>CSV 필수 컬럼 개수</span>
            </div>

            <div className="abnormal-summary-card">
              <p>위험/주의 기록</p>
              <strong>{stats?.total || 0}</strong>
              <span>위험/주의만 저장된 누적 기록</span>
            </div>

            <div className="abnormal-summary-card danger-soft">
              <p>보호자 알림</p>
              <strong>{stats?.guardian_alert_count || 0}</strong>
              <span>즉시 확인 필요 건수</span>
            </div>
          </section>

          <section className="abnormal-middle-grid">
            <div className="abnormal-panel abnormal-upload-panel">
              <div className="abnormal-panel-header">
                <div>
                  <h2>CSV 이상행동 예측 테스트</h2>
                  <p>
                    CSV 또는 Excel 파일을 업로드한 뒤 10초 간격으로 다음
                    데이터를 예측합니다. 기록은 위험/주의만 저장됩니다.
                  </p>
                </div>

                <button
                  className="abnormal-ghost-btn"
                  onClick={handleResetSimulation}
                >
                  처음부터
                </button>
              </div>

              <div className="abnormal-upload-box">
                <div className="abnormal-file-row">
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileChange}
                  />

                  <button
                    className="abnormal-primary-btn"
                    onClick={handleUpload}
                    disabled={uploading}
                  >
                    {uploading ? "업로드 중..." : "파일 업로드"}
                  </button>
                </div>

                <p className="abnormal-upload-help">
                  필요한 컬럼:{" "}
                  {features.length > 0 ? features.join(", ") : "불러오는 중"}
                </p>

                <div className="abnormal-sim-status">
                  <div>
                    <span>업로드 파일</span>
                    <strong>
                      {simulationStatus?.loaded
                        ? simulationStatus.filename
                        : uploadInfo?.filename || "없음"}
                    </strong>
                  </div>

                  <div>
                    <span>전체 행 수</span>
                    <strong>{simulationStatus?.rows ?? "-"}</strong>
                  </div>

                  <div>
                    <span>현재 진행</span>
                    <strong>{progressText}</strong>
                  </div>

                  <div>
                    <span>남은 데이터</span>
                    <strong>{simulationStatus?.remaining ?? "-"}</strong>
                  </div>
                </div>

                <div className="abnormal-action-row">
                  <button
                    className="abnormal-primary-btn full"
                    onClick={handleAutoToggle}
                    disabled={!canRunAuto && !autoRunning}
                  >
                    {autoRunning
                      ? "10초 자동 예측 정지"
                      : "10초 자동 예측 시작"}
                  </button>

                  <button
                    className="abnormal-ghost-btn abnormal-next-btn"
                    onClick={handleNext}
                    disabled={!canRunNext}
                  >
                    {predicting ? "예측 중..." : "다음 10초 실행"}
                  </button>
                </div>
              </div>
            </div>

            <div className="abnormal-panel abnormal-history-panel">
              <div className="abnormal-panel-header">
                <div>
                  <h2>최근 이상행동 기록</h2>
                  <p>
                    위험/주의로 저장된 기록만 표시합니다. 기록을 클릭하면 아래
                    예측 결과 영역에서 상세 내용을 볼 수 있습니다.
                  </p>
                </div>

                <div className="abnormal-button-row">
                  <button className="abnormal-ghost-btn" onClick={loadPageData}>
                    새로고침
                  </button>
                  <button
                    className="abnormal-danger-btn"
                    onClick={handleDeleteHistory}
                  >
                    기록 삭제
                  </button>
                </div>
              </div>

              {history.length === 0 ? (
                <div className="abnormal-empty">
                  저장된 위험/주의 기록이 없습니다.
                </div>
              ) : (
                <>
                  <div className="abnormal-table-wrap abnormal-history-table-wrap">
                    <table className="abnormal-table">
                      <thead>
                        <tr>
                          <th>시간</th>
                          <th>상태</th>
                          <th>점수</th>
                          <th>사유</th>
                          <th>알림</th>
                        </tr>
                      </thead>

                      <tbody>
                        {pagedHistory.map((item, index) => (
                          <tr
                            key={`${item.time}-${currentHistoryPage}-${index}`}
                            className="abnormal-click-row"
                            onClick={() => setSelectedHistoryResult(item)}
                          >
                            <td>{item.time}</td>
                            <td>
                              <span
                                className={`abnormal-mini-state ${getStateClass(
                                  item.state,
                                )}`}
                              >
                                {item.state}
                              </span>
                            </td>
                            <td>{formatValue(item.risk_score)}</td>
                            <td>{item.reason || "-"}</td>
                            <td>
                              {item.guardian_alert
                                ? "보호자 확인 필요"
                                : "알림 없음"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="abnormal-pagination">
                    <button
                      type="button"
                      onClick={handlePrevHistoryPage}
                      disabled={currentHistoryPage <= 1}
                    >
                      {"<"}
                    </button>

                    <strong>
                      {currentHistoryPage} / {totalHistoryPages}
                    </strong>

                    <button
                      type="button"
                      onClick={handleNextHistoryPage}
                      disabled={currentHistoryPage >= totalHistoryPages}
                    >
                      {">"}
                    </button>
                  </div>
                </>
              )}
            </div>
          </section>

          <section className="abnormal-result-row">
            <div className="abnormal-panel abnormal-result-panel">
              <div className="abnormal-panel-header">
                <div>
                  <h2>예측 결과</h2>
                  <p>
                    최신 예측 결과 또는 오른쪽 기록에서 선택한 결과를
                    표시합니다. 수면/식사/외출/정상은 이 영역에만 표시되고
                    기록에는 저장하지 않습니다.
                  </p>
                </div>
              </div>

              {!displayResult ? (
                <div className="abnormal-empty">
                  아직 예측 결과가 없습니다.
                  <br />
                  CSV 파일을 업로드하고 10초 자동 예측 시작을 눌러주세요.
                </div>
              ) : (
                <div className="abnormal-result-layout">
                  <div className="abnormal-result-score-card">
                    <div
                      className={`abnormal-state-badge ${getStateClass(
                        displayResult.state,
                      )}`}
                    >
                      {displayResult.state || "상태 없음"}
                    </div>

                    <div className="abnormal-risk-score-wrap">
                      <div className="abnormal-risk-score">
                        {displayResult.risk_score || 0}
                        <span>점</span>
                      </div>
                      <p>{getRiskText(displayResult.risk_score)} 수준</p>
                    </div>

                    <div className="abnormal-risk-meter">
                      <div
                        className={`abnormal-risk-meter-fill ${getStateClass(
                          displayResult.state,
                        )}`}
                        style={{
                          width: `${Math.min(
                            100,
                            Math.max(0, Number(displayResult.risk_score || 0)),
                          )}%`,
                        }}
                      />
                    </div>

                    <p className="abnormal-guardian-message">
                      {displayResult.guardian_message || ""}
                    </p>
                  </div>

                  <div className="abnormal-result-list">
                    <div>
                      <span>예측 상태</span>
                      <strong>{displayResult.state || "-"}</strong>
                    </div>

                    <div>
                      <span>예측 사유</span>
                      <strong>{displayResult.reason || "-"}</strong>
                    </div>

                    <div>
                      <span>기록 저장</span>
                      <strong>
                        {displayResult.saved_to_history
                          ? "저장됨"
                          : "저장 안 함"}
                      </strong>
                    </div>

                    <div>
                      <span>보호자 알림</span>
                      <strong>
                        {displayResult.guardian_alert
                          ? "발송 필요"
                          : "알림 없음"}
                      </strong>
                    </div>

                    <div>
                      <span>알림 상태</span>
                      <strong>{displayResult.guardian_status || "-"}</strong>
                    </div>

                    <div>
                      <span>시간</span>
                      <strong>{displayResult.time || "-"}</strong>
                    </div>

                    <div>
                      <span>출처</span>
                      <strong>{displayResult.source || "-"}</strong>
                    </div>

                    <div>
                      <span>저장 정책</span>
                      <strong>위험/주의만 저장</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="abnormal-result-empty-right" />
          </section>

          <SensorInfoSection sensor={displayResult?.sensor} />
        </>
      )}
    </main>
  );
}

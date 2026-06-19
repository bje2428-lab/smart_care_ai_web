import React from "react";
import axios from "axios";

import "./FallDashboard.css";

const API_URLS = (
  import.meta.env.VITE_API_URLS ||
  import.meta.env.VITE_API_URL ||
  ""
)
  .split(",")
  .map((url) => url.trim())
  .filter(Boolean);

let activeApiBaseUrl = null;

async function getActiveApiBaseUrl() {
  if (activeApiBaseUrl) {
    return activeApiBaseUrl;
  }

  if (API_URLS.length === 0) {
    throw new Error(".env에 VITE_API_URLS가 설정되어 있지 않습니다.");
  }

  for (const url of API_URLS) {
    try {
      const res = await axios.get(`${url}/`, {
        timeout: 2000,
      });

      if (res.status === 200) {
        activeApiBaseUrl = url;
        console.log("연결된 백엔드:", activeApiBaseUrl);
        return activeApiBaseUrl;
      }
    } catch (err) {
      console.log(`${url} 연결 실패`);
    }
  }

  throw new Error("사용 가능한 백엔드 서버가 없습니다.");
}

function resetActiveApiBaseUrl() {
  activeApiBaseUrl = null;
}

async function apiGet(path) {
  const baseUrl = await getActiveApiBaseUrl();
  return axios.get(`${baseUrl}${path}`);
}

async function apiPostForm(path, formData) {
  const baseUrl = await getActiveApiBaseUrl();

  return axios.post(`${baseUrl}${path}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
}

async function apiDelete(path) {
  const baseUrl = await getActiveApiBaseUrl();
  return axios.delete(`${baseUrl}${path}`);
}

const DEFAULT_SENSOR_FEATURES = [
  {
    name: "x",
    label: "좌우 위치",
    meaning: "레이더 기준 사람/물체가 좌우로 얼마나 이동했는지 나타내는 값",
    mean: "-",
    min: "-",
    max: "-",
    range: 0,
  },
  {
    name: "y",
    label: "전후 거리",
    meaning: "레이더와 대상 사이의 앞뒤 거리 변화를 나타내는 값",
    mean: "-",
    min: "-",
    max: "-",
    range: 0,
  },
  {
    name: "z",
    label: "높이",
    meaning:
      "대상의 높이 변화. 낙상, 앉기, 눕기처럼 자세가 낮아질 때 크게 변할 수 있음",
    mean: "-",
    min: "-",
    max: "-",
    range: 0,
  },
  {
    name: "v",
    label: "속도",
    meaning:
      "대상의 움직임 속도. 순간적으로 빠른 움직임이 있으면 값이 커질 수 있음",
    mean: "-",
    min: "-",
    max: "-",
    range: 0,
  },
];

function StatusBadge({ status }) {
  const cls =
    status === "Fall Alert"
      ? "badge danger"
      : status === "Exception"
        ? "badge warning"
        : status === "Normal"
          ? "badge normal"
          : "badge";

  return <span className={cls}>{status || "-"}</span>;
}

function AlertText({ alert }) {
  return alert ? (
    <span className="alert-true">알림 발생</span>
  ) : (
    <span className="alert-false">알림 없음</span>
  );
}

function toSafeNumber(value) {
  if (value === null || value === undefined) return 0;

  if (typeof value === "string") {
    const cleaned = value.replace("%", "").trim();
    const num = Number(cleaned);
    if (!Number.isFinite(num)) return 0;
    return value.includes("%") ? num / 100 : num;
  }

  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
}

function normalizeProbability(value) {
  const num = toSafeNumber(value);
  return num > 1 ? num / 100 : num;
}

function ProbabilityChart({ value = 0, status }) {
  const prob = normalizeProbability(value);
  const percent = Math.round(prob * 100);
  const safePercent = Math.max(0, Math.min(100, percent));

  let levelText = "낮음";
  if (safePercent >= 85) levelText = "매우 높음";
  else if (safePercent >= 60) levelText = "높음";
  else if (safePercent >= 30) levelText = "주의";

  return (
    <div className="chart-card">
      <div className="chart-title-row">
        <div>
          <h3>낙상 확률 그래프</h3>
          <p>모델이 업로드한 CSV를 낙상으로 판단한 확률입니다.</p>
        </div>
        <StatusBadge status={status} />
      </div>

      <div className="probability-graph">
        <div className="probability-track">
          <div
            className={`probability-fill ${
              safePercent >= 85
                ? "danger-fill"
                : safePercent >= 60
                  ? "warning-fill"
                  : "normal-fill"
            }`}
            style={{ width: `${safePercent}%` }}
          />
        </div>
        <div className="probability-info">
          <strong>{safePercent}%</strong>
          <span>위험도: {levelText}</span>
        </div>
      </div>

      <div className="scale-row">
        <span>0%</span>
        <span>30%</span>
        <span>60%</span>
        <span>85%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function DecisionEvidenceChart({ result }) {
  const fallProb = normalizeProbability(result?.fall_prob);
  const heightDrop = toSafeNumber(result?.height_drop);
  const speedMax = toSafeNumber(result?.speed_max);
  const movementAfter = toSafeNumber(result?.movement_after);

  const getLevel = (type, value) => {
    if (!result) return "normal";

    if (type === "fallProb") {
      if (value >= 0.7) return "danger";
      if (value >= 0.4) return "warning";
      return "normal";
    }

    if (type === "heightDrop") {
      if (value >= 0.8) return "danger";
      if (value >= 0.4) return "warning";
      return "normal";
    }

    if (type === "speedMax") {
      if (value >= 1.2) return "danger";
      if (value >= 0.6) return "warning";
      return "normal";
    }

    if (type === "movementAfter") {
      if (value <= 0.2) return "danger";
      if (value <= 0.5) return "warning";
      return "normal";
    }

    return "normal";
  };

  const rows = [
    {
      key: "fallProb",
      label: "낙상 확률",
      desc: "모델이 낙상으로 판단한 확률",
      value: fallProb,
      max: 1,
      displayValue: result ? `${Math.round(fallProb * 100)}%` : "-",
      level: getLevel("fallProb", fallProb),
    },
    {
      key: "heightDrop",
      label: "높이 변화",
      desc: "서 있던 대상이 아래로 낮아진 정도",
      value: heightDrop,
      max: 1.5,
      displayValue: result ? heightDrop.toFixed(4) : "-",
      level: getLevel("heightDrop", heightDrop),
    },
    {
      key: "speedMax",
      label: "최대 속도",
      desc: "순간적으로 빠르게 움직였는지",
      value: speedMax,
      max: 2,
      displayValue: result ? speedMax.toFixed(4) : "-",
      level: getLevel("speedMax", speedMax),
    },
    {
      key: "movementAfter",
      label: "이후 움직임",
      desc: "낙상 이후 움직임이 적은지",
      value: movementAfter,
      max: 1,
      displayValue: result ? movementAfter.toFixed(4) : "-",
      level: getLevel("movementAfter", movementAfter),
    },
  ];

  const getDecisionReason = () => {
    if (!result) {
      return {
        title: "아직 예측 전입니다.",
        reasons: ["CSV 파일을 업로드하면 낙상 판단 근거가 표시됩니다."],
      };
    }

    const reasons = [];

    if (fallProb >= 0.7) {
      reasons.push("모델의 낙상 확률이 높게 나왔습니다.");
    } else if (fallProb >= 0.4) {
      reasons.push("모델의 낙상 확률이 중간 수준이라 주의가 필요합니다.");
    } else {
      reasons.push("모델의 낙상 확률이 낮게 나왔습니다.");
    }

    if (heightDrop >= 0.8) {
      reasons.push(
        "높이 변화가 커서 사람이 급격히 낮아진 패턴이 감지되었습니다.",
      );
    } else if (heightDrop >= 0.4) {
      reasons.push(
        "높이 변화가 일부 감지되었지만 낙상으로 확정하기에는 애매합니다.",
      );
    } else {
      reasons.push("높이 변화가 크지 않아 낙상 가능성이 낮습니다.");
    }

    if (speedMax >= 1.2) {
      reasons.push("순간 속도가 높아 갑작스러운 움직임이 감지되었습니다.");
    } else if (speedMax >= 0.6) {
      reasons.push(
        "움직임 속도가 어느 정도 있었지만 매우 급격한 수준은 아닙니다.",
      );
    } else {
      reasons.push("움직임 속도가 크지 않았습니다.");
    }

    if (movementAfter <= 0.2) {
      reasons.push(
        "이후 움직임이 적어 쓰러진 뒤 움직이지 않는 패턴과 유사합니다.",
      );
    } else if (movementAfter <= 0.5) {
      reasons.push(
        "이후 움직임이 적은 편이라 예외 또는 주의 상황으로 볼 수 있습니다.",
      );
    } else {
      reasons.push(
        "이후 움직임이 계속 감지되어 낙상 후 정지 상태로 보기는 어렵습니다.",
      );
    }

    if (result.status === "Fall Alert") {
      return {
        title: "낙상으로 판단한 이유",
        reasons,
      };
    }

    if (result.status === "Exception") {
      return {
        title: "예외행동으로 판단한 이유",
        reasons: [
          ...reasons,
          "앉기, 눕기, 물건 줍기처럼 낙상과 비슷한 움직임일 수 있어 예외행동으로 분류했습니다.",
        ],
      };
    }

    if (result.status === "Normal") {
      return {
        title: "정상으로 판단한 이유",
        reasons,
      };
    }

    return {
      title: "판단 이유",
      reasons,
    };
  };

  const decision = getDecisionReason();

  return (
    <div className="chart-card">
      <h3>낙상 판단 근거 그래프</h3>
      <p>
        낙상 확률, 높이 변화, 최대 속도, 이후 움직임을 기준으로 모델이 왜
        낙상/정상/예외로 판단했는지 보여줍니다.
      </p>

      <div className="bar-chart">
        {rows.map((item) => {
          const width = result
            ? Math.max(
                4,
                Math.min(100, (Number(item.value || 0) / item.max) * 100),
              )
            : 4;

          return (
            <div className="bar-row" key={item.key}>
              <div className="bar-label">
                <strong>{item.label}</strong>
                <span>{item.desc}</span>
              </div>

              <div className="bar-track">
                <div
                  className={`bar-fill evidence-${item.level}`}
                  style={{ width: `${width}%` }}
                />
              </div>

              <div className="bar-value">{item.displayValue}</div>
            </div>
          );
        })}
      </div>

      <div className="evidence-summary">
        <h4>{decision.title}</h4>

        {result && (
          <p className="decision-status">
            최종 판단: <strong>{result.status}</strong>
          </p>
        )}

        <ul>
          {decision.reasons.map((reason, index) => (
            <li key={index}>{reason}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function RecentEventChart({ events }) {
  const recent = (events || []).slice(0, 8).reverse();

  if (recent.length === 0) {
    return (
      <div className="chart-card">
        <h3>최근 낙상 알림 확률 그래프</h3>
        <p className="empty-chart">저장된 낙상 알림 데이터가 없습니다.</p>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <h3>최근 낙상 알림 확률 그래프</h3>
      <p>저장된 최근 Fall Alert 이벤트의 낙상 확률입니다.</p>

      <div className="mini-column-chart">
        {recent.map((event) => {
          const prob = normalizeProbability(event.fall_prob);
          const percent = Math.round(prob * 100);
          const height = Math.max(5, Math.min(100, percent));

          return (
            <div className="mini-column-item" key={event._id}>
              <div className="mini-column-wrap">
                <div className="mini-column" style={{ height: `${height}%` }} />
              </div>
              <span>{percent}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SensorFeatureTable({ rows }) {
  const data = rows && rows.length > 0 ? rows : DEFAULT_SENSOR_FEATURES;

  return (
    <div className="sensor-table-card">
      <h3>mmWave 주요 값 설명</h3>
      <p className="desc">
        CSV에 포함된 x, y, z, v 값을 요약해서 보여줍니다. 예측 후에는 업로드한
        파일의 평균, 최솟값, 최댓값, 변화범위가 함께 표시됩니다.
      </p>

      <div className="table-wrap sensor-wrap">
        <table>
          <thead>
            <tr>
              <th>값</th>
              <th>의미</th>
              <th>설명</th>
              <th>평균</th>
              <th>최소</th>
              <th>최대</th>
              <th>범위</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr key={item.name}>
                <td className="sensor-name">{item.name}</td>
                <td>{item.label}</td>
                <td>{item.meaning}</td>
                <td>{item.mean}</td>
                <td>{item.min}</td>
                <td>{item.max}</td>
                <td>{item.range}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FallDashboard() {
  const [serverInfo, setServerInfo] = React.useState(null);
  const [stats, setStats] = React.useState(null);
  const [events, setEvents] = React.useState([]);
  const [selectedFile, setSelectedFile] = React.useState(null);
  const [predictResult, setPredictResult] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [message, setMessage] = React.useState("");

  const loadServerInfo = async () => {
    try {
      const res = await apiGet("/");
      setServerInfo(res.data);
    } catch (err) {
      setServerInfo(null);
      setMessage(
        "FastAPI 서버에 연결할 수 없습니다. 백엔드가 켜져 있는지 확인하세요.",
      );
    }
  };

  const loadStats = async () => {
    try {
      const res = await apiGet("/stats");
      setStats(res.data);
    } catch (err) {
      setStats(null);
    }
  };

  const loadEvents = async () => {
    try {
      const res = await apiGet("/events?limit=30");
      setEvents(res.data.events || []);
    } catch (err) {
      setEvents([]);
    }
  };

  const refreshAll = async () => {
    resetActiveApiBaseUrl();
    await loadServerInfo();
    await loadStats();
    await loadEvents();
  };

  React.useEffect(() => {
    refreshAll();
  }, []);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setPredictResult(null);
    setMessage("");
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setMessage("먼저 CSV 파일을 선택하세요.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);
    setMessage("");

    try {
      const res = await apiPostForm("/predict", formData);

      setPredictResult(res.data);

      if (res.data.status === "Fall Alert") {
        setMessage("낙상 알림이 감지되어 저장했습니다.");
      } else if (res.data.status === "Exception") {
        setMessage("예외행동으로 처리되었습니다. DB에는 저장하지 않습니다.");
      } else if (res.data.status === "Normal") {
        setMessage("정상 행동으로 판단되었습니다. DB에는 저장하지 않습니다.");
      } else {
        setMessage(res.data.message || "처리 완료");
      }

      await loadStats();
      await loadEvents();
    } catch (err) {
      setMessage("예측 요청 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteEvents = async () => {
    const ok = window.confirm("저장된 낙상 알림 로그를 모두 삭제할까요?");
    if (!ok) return;

    try {
      await apiDelete("/events");
      await loadStats();
      await loadEvents();
      setMessage("이벤트 로그를 삭제했습니다.");
    } catch (err) {
      setMessage("이벤트 삭제 중 오류가 발생했습니다.");
    }
  };

  return (
    <div className="page">
      <header className="dashboard-header">
        <div>
          <h1>Smart Care AI</h1>
          <p>mmWave 기반 독거노인 낙상 감지 관제 대시보드</p>
        </div>
        <button className="refresh-btn" onClick={refreshAll}>
          새로고침
        </button>
      </header>

      <section className="status-panel">
        <div className="server-card">
          <h2>서버 상태</h2>
          <div className="server-grid">
            <div>
              <span>FastAPI</span>
              <strong>{serverInfo ? "연결됨" : "연결 안 됨"}</strong>
            </div>
            <div>
              <span>모델 파일</span>
              <strong>{serverInfo?.model_exists ? "있음" : "없음"}</strong>
            </div>
            <div>
              <span>MongoDB</span>
              <strong>
                {serverInfo?.mongo_connected ? "연결됨" : "연결 안 됨"}
              </strong>
            </div>
          </div>
          <p className="policy-text">
            저장 정책: <strong>Fall Alert만 저장</strong>, Exception/Normal은
            저장하지 않음
          </p>
        </div>

        <div className="stats">
          <div className="stat-card">
            <span>저장된 전체 로그</span>
            <strong>{stats?.total_events ?? 0}</strong>
          </div>

          <div className="stat-card danger-card">
            <span>낙상 알림</span>
            <strong>{stats?.fall_alert_count ?? 0}</strong>
          </div>

          <div className="stat-card illustration-card">
            <span>스마트 돌봄 캐릭터</span>

            <div className="elder-illustration">
              <div className="elder elder-grandma">
                <div className="hair"></div>
                <div className="face">
                  <div className="eyes"></div>
                  <div className="smile"></div>
                  <div className="blush blush-left"></div>
                  <div className="blush blush-right"></div>
                </div>
                <div className="body"></div>
              </div>

              <div className="elder elder-grandpa">
                <div className="hair"></div>
                <div className="face">
                  <div className="glasses">
                    <span></span>
                    <span></span>
                  </div>
                  <div className="eyes"></div>
                  <div className="smile"></div>
                  <div className="blush blush-left"></div>
                  <div className="blush blush-right"></div>
                </div>
                <div className="body"></div>
              </div>
            </div>

            <p className="illustration-text">
              어르신의 안전한 일상을 함께 지켜봐요
            </p>
          </div>
        </div>
      </section>

      <main className="main-grid">
        <section className="card upload-card">
          <h2>CSV 낙상 예측 테스트</h2>
          <p className="desc">
            mmWave CSV 파일을 업로드하면 낙상 여부를 예측합니다. Fall Alert인
            경우에만 저장합니다.
          </p>

          <div className="upload-box">
            <input type="file" accept=".csv" onChange={handleFileChange} />
            <button onClick={handlePredict} disabled={loading}>
              {loading ? "분석 중..." : "예측 실행"}
            </button>
          </div>

          {selectedFile && (
            <p className="selected-file">선택 파일: {selectedFile.name}</p>
          )}

          {message && <div className="message">{message}</div>}

          {predictResult && (
            <div className="result-box">
              <div className="result-header">
                <StatusBadge status={predictResult.status} />
                <AlertText alert={predictResult.alert} />
              </div>

              <p className="result-message">{predictResult.message}</p>

              <div className="result-grid">
                <div>
                  <span>파일명</span>
                  <strong>{predictResult.file_name}</strong>
                </div>
                <div>
                  <span>낙상 확률</span>
                  <strong>{predictResult.fall_prob}</strong>
                </div>
                <div>
                  <span>속도 최댓값</span>
                  <strong>{predictResult.speed_max}</strong>
                </div>
                <div>
                  <span>높이 변화</span>
                  <strong>{predictResult.height_drop}</strong>
                </div>
                <div>
                  <span>이후 움직임</span>
                  <strong>{predictResult.movement_after}</strong>
                </div>
                <div>
                  <span>DB 저장</span>
                  <strong>
                    {predictResult.db_saved ? "저장됨" : "저장 안 함"}
                  </strong>
                </div>
              </div>

              {predictResult.db_message && (
                <p className="db-message">{predictResult.db_message}</p>
              )}
            </div>
          )}

          <ProbabilityChart
            value={predictResult?.fall_prob || 0}
            status={predictResult?.status || "Normal"}
          />

          <DecisionEvidenceChart result={predictResult} />

          <SensorFeatureTable rows={predictResult?.sensor_features} />
        </section>

        <section className="card events-card">
          <div className="section-title-row">
            <div>
              <h2>최근 낙상 알림 로그</h2>
              <p className="desc">
                저장된 Fall Alert 이벤트만 표시합니다. Exception은 저장하지
                않습니다.
              </p>
            </div>
            <button className="delete-btn" onClick={handleDeleteEvents}>
              로그 삭제
            </button>
          </div>

          <RecentEventChart events={events} />

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>시간</th>
                  <th>상태</th>
                  <th>알림</th>
                  <th>파일명</th>
                  <th>확률</th>
                  <th>DB ID</th>
                </tr>
              </thead>
              <tbody>
                {events.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="empty">
                      저장된 낙상 알림이 없습니다.
                    </td>
                  </tr>
                ) : (
                  events.map((event) => (
                    <tr key={event._id}>
                      <td>{event.created_at}</td>
                      <td>
                        <StatusBadge status={event.status} />
                      </td>
                      <td>
                        <AlertText alert={event.alert} />
                      </td>
                      <td>{event.file_name}</td>
                      <td>{event.fall_prob}</td>
                      <td>{String(event._id).slice(-6)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default FallDashboard;

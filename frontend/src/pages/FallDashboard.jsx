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

const EVENT_PAGE_SIZE = 5;

let activeApiBaseUrl = null;

async function getActiveApiBaseUrl() {
  if (activeApiBaseUrl) return activeApiBaseUrl;

  if (API_URLS.length === 0) {
    throw new Error(
      ".env에 VITE_API_URLS 또는 VITE_API_URL이 설정되어 있지 않습니다.",
    );
  }

  for (const url of API_URLS) {
    try {
      const res = await axios.get(`${url}/`, { timeout: 2000 });

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

function getEventId(event) {
  return String(event?._id || event?.id || event?.file_name || "");
}

function getDisplayStatus(status) {
  if (status === "Exception") return "Normal";
  return status || "-";
}

function getDisplayStatusText(status) {
  if (status === "Exception") return "Normal";
  return status || "-";
}

function getDisplayResultMessage(result) {
  if (!result) return "";

  if (result.status === "Fall Alert") {
    return result.message || "낙상 알림으로 판단되었습니다.";
  }

  return "정상 행동으로 판단되었습니다. 낙상 알림 기준에 도달하지 않아 DB에는 저장하지 않습니다.";
}

function StatusBadge({ status }) {
  const displayStatus = getDisplayStatus(status);

  const cls =
    displayStatus === "Fall Alert"
      ? "badge danger"
      : displayStatus === "Normal"
        ? "badge normal"
        : "badge";

  return <span className={cls}>{displayStatus}</span>;
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

function getModelFallProbability(result) {
  if (!result) return 0;

  const value =
    result.raw_model_fall_prob !== undefined &&
    result.raw_model_fall_prob !== null
      ? result.raw_model_fall_prob
      : result.fall_prob;

  return normalizeProbability(value);
}

function getDisplayFallProbability(result) {
  if (!result) return 0;
  return getModelFallProbability(result);
}

function getDisplayFallPercent(result) {
  return Math.round(getDisplayFallProbability(result) * 100);
}

function getRiskLevelText(percent) {
  if (percent >= 85) return "매우 높음";
  if (percent >= 60) return "높음";
  if (percent >= 30) return "주의";
  return "낮음";
}

function formatMeter(value) {
  return `${toSafeNumber(value).toFixed(4)} m`;
}

function formatSpeed(value) {
  return `${toSafeNumber(value).toFixed(4)} m/s`;
}

function formatMovementCm(value) {
  return `${(toSafeNumber(value) * 100).toFixed(2)} cm`;
}

function formatDateTime(value) {
  if (!value) return "-";
  return String(value).replace("T", " ").replace(".000Z", "").slice(0, 19);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/* =========================================================
   CSV 파싱 + 프레임별 낙상 점수 계산
========================================================= */

function parseCsvLine(line) {
  const result = [];
  let current = "";
  let insideQuote = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      insideQuote = !insideQuote;
    } else if (char === "," && !insideQuote) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}

function downSampleRows(rows, maxPoints = 160) {
  if (rows.length <= maxPoints) return rows;

  const step = Math.ceil(rows.length / maxPoints);
  return rows.filter((_, index) => index % step === 0);
}

function getDistance3d(a, b) {
  if (!a || !b) return 0;

  const dx = toSafeNumber(b.x) - toSafeNumber(a.x);
  const dy = toSafeNumber(b.y) - toSafeNumber(a.y);
  const dz = toSafeNumber(b.z) - toSafeNumber(a.z);

  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function getAfterMovement(rows, startIndex, frameCount = 12) {
  let total = 0;
  const endIndex = Math.min(rows.length - 1, startIndex + frameCount);

  for (let i = startIndex; i < endIndex; i += 1) {
    total += getDistance3d(rows[i], rows[i + 1]);
  }

  return total;
}

function movingAverageValues(values, windowSize = 5) {
  return values.map((_, index) => {
    const start = Math.max(0, index - Math.floor(windowSize / 2));
    const end = Math.min(values.length - 1, index + Math.floor(windowSize / 2));

    let sum = 0;
    let count = 0;

    for (let i = start; i <= end; i += 1) {
      sum += toSafeNumber(values[i]);
      count += 1;
    }

    return count > 0 ? sum / count : 0;
  });
}

function buildFallScoreRows(rows) {
  if (!rows || rows.length < 8) return [];

  const windowSize = Math.min(8, Math.max(3, Math.floor(rows.length / 14)));

  const candidates = rows.map((row, index) => {
    const next = rows[Math.min(rows.length - 1, index + windowSize)];

    const zDrop = Math.max(0, toSafeNumber(row.z) - toSafeNumber(next.z));
    const speed = Math.abs(toSafeNumber(row.v));
    const afterMovement = getAfterMovement(rows, index, 12);

    return {
      index,
      frame: row.frame,
      zDrop,
      speed,
      afterMovement,
    };
  });

  const maxDrop = Math.max(...candidates.map((item) => item.zDrop), 0.0001);
  const maxSpeed = Math.max(...candidates.map((item) => item.speed), 0.0001);
  const maxAfterMovement = Math.max(
    ...candidates.map((item) => item.afterMovement),
    0.0001,
  );

  const rawScoreRows = candidates.map((item) => {
    const heightDropScore = clamp(item.zDrop / maxDrop, 0, 1);
    const speedScore = clamp(item.speed / maxSpeed, 0, 1);
    const stillnessScore =
      1 - clamp(item.afterMovement / maxAfterMovement, 0, 1);

    const rawFallScore =
      heightDropScore * 50 + speedScore * 30 + stillnessScore * 20;

    return {
      index: item.index,
      frame: item.frame,
      rawFallScore,
      heightDropScore: heightDropScore * 100,
      speedScore: speedScore * 100,
      stillnessScore: stillnessScore * 100,
      zDrop: item.zDrop,
      speed: item.speed,
      afterMovement: item.afterMovement,
    };
  });

  const smoothedRawScores = movingAverageValues(
    rawScoreRows.map((item) => item.rawFallScore),
    5,
  );

  const maxSmoothedScore = Math.max(...smoothedRawScores, 0.0001);

  return rawScoreRows.map((item, index) => ({
    ...item,
    rawFallScore: clamp(smoothedRawScores[index], 0, 100),
    fallScore: clamp(
      (smoothedRawScores[index] / maxSmoothedScore) * 100,
      0,
      100,
    ),
  }));
}

function parseFallScoreFromCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return {
      rows: [],
    };
  }

  const headers = parseCsvLine(lines[0]).map((h) => h.trim());

  const findIndex = (name) =>
    headers.findIndex((header) => header.toLowerCase() === name.toLowerCase());

  const frameIndex = findIndex("frame");
  const xIndex = findIndex("x");
  const yIndex = findIndex("y");
  const zIndex = findIndex("z");
  const vIndex = findIndex("v");

  if (xIndex === -1 || yIndex === -1 || zIndex === -1 || vIndex === -1) {
    return {
      rows: [],
    };
  }

  const rawRows = lines.slice(1).map((line, index) => {
    const cols = parseCsvLine(line);

    return {
      frame:
        frameIndex >= 0 && Number.isFinite(Number(cols[frameIndex]))
          ? Number(cols[frameIndex])
          : index + 1,
      x: toSafeNumber(cols[xIndex]),
      y: toSafeNumber(cols[yIndex]),
      z: toSafeNumber(cols[zIndex]),
      v: toSafeNumber(cols[vIndex]),
    };
  });

  const scoreRows = buildFallScoreRows(rawRows);

  return {
    rows: downSampleRows(scoreRows),
  };
}

/* =========================================================
   프레임별 그래프 계산
========================================================= */

const HEIGHT_RISK_STANDARD = 0.3;
const SPEED_RISK_STANDARD = 0.35;
const STILLNESS_RISK_STANDARD = 0.2;

function getRuleBasedFallProbability(result) {
  if (!result) return 0;

  const heightDrop = toSafeNumber(result.height_drop);
  const speedMax = toSafeNumber(result.speed_max);
  const movementAfter = toSafeNumber(result.movement_after);

  const hasEvidence = heightDrop > 0 || speedMax > 0 || movementAfter > 0;

  if (!hasEvidence) return 0;

  const heightScore = clamp(heightDrop / HEIGHT_RISK_STANDARD, 0, 1);
  const speedScore = clamp(speedMax / SPEED_RISK_STANDARD, 0, 1);

  let stillnessScore = 0;

  if (movementAfter > 0 && movementAfter <= STILLNESS_RISK_STANDARD) {
    stillnessScore = 1;
  } else if (movementAfter > STILLNESS_RISK_STANDARD) {
    stillnessScore = clamp((0.5 - movementAfter) / 0.3, 0, 1);
  }

  const ruleScore = heightScore * 0.5 + speedScore * 0.3 + stillnessScore * 0.2;

  return clamp(ruleScore, 0, 1);
}

function getXByIndex(index, dataLength, width, padding) {
  const usableWidth = width - padding.left - padding.right;
  return padding.left + (index / Math.max(1, dataLength - 1)) * usableWidth;
}

function getYByScore(score, height, padding) {
  const usableHeight = height - padding.top - padding.bottom;
  return padding.top + (1 - clamp(score, 0, 100) / 100) * usableHeight;
}

function buildDisplayScoreData(data, result, threshold = 70) {
  const modelFallPercent = getDisplayFallPercent(result);
  const ruleFallPercent = Math.round(getRuleBasedFallProbability(result) * 100);
  const isFallAlert = result?.status === "Fall Alert";

  let targetPeak = 0;

  if (isFallAlert) {
    targetPeak = Math.max(modelFallPercent, threshold + 12);
  } else {
    const sensorPeak = ruleFallPercent > 0 ? ruleFallPercent : modelFallPercent;
    targetPeak = Math.min(sensorPeak, threshold - 12);

    if (targetPeak < 20 && data.length > 0) {
      targetPeak = 20;
    }
  }

  const safeTargetPeak = clamp(targetPeak, 0, 100);

  const maxScore = Math.max(
    ...data.map((item) => toSafeNumber(item.fallScore)),
    0.0001,
  );

  return data.map((item) => ({
    ...item,
    displayScore: clamp(
      (toSafeNumber(item.fallScore) / maxScore) * safeTargetPeak,
      0,
      100,
    ),
  }));
}

function getScoreSvgPoints(data, width, height, padding) {
  return data
    .map((item, index) => {
      const x = getXByIndex(index, data.length, width, padding);
      const y = getYByScore(item.displayScore, height, padding);

      return `${x},${y}`;
    })
    .join(" ");
}

function detectDisplayFallZone(displayData, threshold = 70) {
  if (!displayData || displayData.length === 0) return null;

  const overIndexes = displayData
    .map((item, index) => ({
      item,
      index,
    }))
    .filter(({ item }) => item.displayScore >= threshold);

  if (overIndexes.length === 0) return null;

  let best = overIndexes[0];

  overIndexes.forEach((candidate) => {
    if (candidate.item.displayScore > best.item.displayScore) {
      best = candidate;
    }
  });

  let left = best.index;
  let right = best.index;

  while (left > 0 && displayData[left - 1].displayScore >= threshold) {
    left -= 1;
  }

  while (
    right < displayData.length - 1 &&
    displayData[right + 1].displayScore >= threshold
  ) {
    right += 1;
  }

  return {
    startFrame: displayData[left].frame,
    centerFrame: best.item.frame,
    endFrame: displayData[right].frame,
    peakScore: Math.round(best.item.displayScore),
    startIndex: left,
    centerIndex: best.index,
    endIndex: right,
  };
}

/* =========================================================
   차트 컴포넌트
========================================================= */

function ProbabilityChart({ result }) {
  if (!result) return null;

  const safePercent = Math.max(0, Math.min(100, getDisplayFallPercent(result)));
  const levelText = getRiskLevelText(safePercent);

  return (
    <div className="chart-card">
      <div className="chart-title-row">
        <div>
          <h3>낙상 위험도</h3>
          <p>
            AI 모델이 업로드된 센서 데이터를 분석해 낙상 위험 수준을 판단한
            결과입니다.
          </p>
        </div>
        <StatusBadge status={result.status} />
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
          <span>낙상 위험도: {levelText}</span>
        </div>
      </div>

      <div className="scale-row">
        <span>0%</span>
        <span>30%</span>
        <span>60%</span>
        <span>85%</span>
        <span>100%</span>
      </div>

      {result.status !== "Fall Alert" && (
        <p className="db-message">
          정상 행동으로 분류되었습니다. 낙상 알림이 아니므로 DB에는 저장하지
          않습니다.
        </p>
      )}
    </div>
  );
}

function FallScoreChart({ data, result }) {
  if (!result) return null;

  if (!data || data.length === 0) {
    return (
      <div className="chart-card">
        <h3>프레임별 낙상 판정 그래프</h3>
        <p className="empty-chart">
          과거 로그는 원본 CSV 프레임 데이터가 저장되어 있지 않아 프레임별
          그래프를 표시할 수 없습니다.
        </p>
      </div>
    );
  }

  const threshold = 70;
  const displayData = buildDisplayScoreData(data, result, threshold);
  const displayZone = detectDisplayFallZone(displayData, threshold);

  const width = 760;
  const height = 280;
  const padding = {
    top: 32,
    right: 30,
    bottom: 42,
    left: 54,
  };

  const firstFrame = displayData[0]?.frame ?? 1;
  const lastFrame =
    displayData[displayData.length - 1]?.frame ?? displayData.length;

  const modelFallPercent = getDisplayFallPercent(result);
  const isFallAlert = result.status === "Fall Alert";
  const showZone = isFallAlert && displayZone;
  const thresholdY = getYByScore(threshold, height, padding);

  const zoneX1 = showZone
    ? getXByIndex(displayZone.startIndex, displayData.length, width, padding)
    : 0;
  const zoneX2 = showZone
    ? getXByIndex(displayZone.endIndex, displayData.length, width, padding)
    : 0;
  const centerX = showZone
    ? getXByIndex(displayZone.centerIndex, displayData.length, width, padding)
    : 0;

  const maxDisplayScore = Math.round(
    Math.max(...displayData.map((item) => item.displayScore), 0),
  );

  return (
    <div className="chart-card">
      <div className="chart-title-row">
        <div>
          <h3>프레임별 낙상 판정 그래프</h3>
          <p>
            프레임별 센서 패턴을 낙상 판정 점수로 변환해 보여줍니다. 점수가
            기준선 70%를 넘으면 낙상 의심 구간으로 표시합니다.
          </p>
        </div>

        {showZone ? (
          <span className="fall-zone-chip">기준선 초과</span>
        ) : (
          <span className="normal-zone-chip">기준선 미만</span>
        )}
      </div>

      <div className="line-chart-wrap">
        <svg
          className="fall-score-svg"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="프레임별 낙상 판정 그래프"
        >
          {showZone && (
            <>
              <rect
                x={Math.min(zoneX1, zoneX2)}
                y={padding.top}
                width={Math.max(8, Math.abs(zoneX2 - zoneX1))}
                height={height - padding.top - padding.bottom}
                className="fall-zone-area"
              />

              <line
                x1={centerX}
                y1={padding.top}
                x2={centerX}
                y2={height - padding.bottom}
                className="fall-zone-center-line"
              />

              <text
                x={centerX}
                y={padding.top - 10}
                textAnchor="middle"
                className="fall-zone-text"
              >
                낙상 의심
              </text>
            </>
          )}

          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={height - padding.bottom}
            className="axis-line"
          />

          <line
            x1={padding.left}
            y1={height - padding.bottom}
            x2={width - padding.right}
            y2={height - padding.bottom}
            className="axis-line"
          />

          <line
            x1={padding.left}
            y1={padding.top}
            x2={width - padding.right}
            y2={padding.top}
            className="grid-line"
          />

          <line
            x1={padding.left}
            y1={height / 2}
            x2={width - padding.right}
            y2={height / 2}
            className="grid-line"
          />

          <line
            x1={padding.left}
            y1={thresholdY}
            x2={width - padding.right}
            y2={thresholdY}
            className="fall-threshold-line"
          />

          <text
            x={padding.left + 8}
            y={thresholdY - 8}
            className="fall-threshold-text"
          >
            낙상 기준선 70%
          </text>

          <polyline
            points={getScoreSvgPoints(displayData, width, height, padding)}
            className="fall-score-line"
          />

          <text x={padding.left} y={height - 12} className="axis-text">
            frame {firstFrame}
          </text>

          <text
            x={width - padding.right}
            y={height - 12}
            textAnchor="end"
            className="axis-text"
          >
            frame {lastFrame}
          </text>

          <text x={10} y={padding.top + 4} className="axis-text">
            100
          </text>

          <text x={16} y={height - padding.bottom} className="axis-text">
            0
          </text>
        </svg>
      </div>

      <div className="fall-score-legend">
        <span className="legend-score">낙상 판정 점수</span>
        <span className="legend-threshold">낙상 기준선 70%</span>
        {showZone && <span className="legend-fall">낙상 의심 구간</span>}
      </div>

      <div className="fall-metric-grid">
        <div>
          <span>모델 낙상 위험도</span>
          <strong>{modelFallPercent}%</strong>
          <p>최종 AI 판단 기준</p>
        </div>

        <div>
          <span>그래프 최고점</span>
          <strong>{maxDisplayScore}%</strong>
          <p>{maxDisplayScore >= threshold ? "기준선 초과" : "기준선 미만"}</p>
        </div>

        <div>
          <span>낙상 기준선</span>
          <strong>{threshold}%</strong>
          <p>넘으면 낙상 의심</p>
        </div>

        <div>
          <span>최종 판단</span>
          <strong>{getDisplayStatusText(result.status)}</strong>
          <p>{isFallAlert ? "Fall Alert 판단" : "정상 행동 판단"}</p>
        </div>
      </div>

      {showZone ? (
        <p className="fall-zone-note">
          그래프의 낙상 판정 점수가 기준선 70%를 초과했기 때문에 frame{" "}
          {displayZone.startFrame} ~ {displayZone.endFrame} 구간을 낙상 의심
          구간으로 표시했습니다. 중심 프레임은 frame {displayZone.centerFrame}
          입니다.
        </p>
      ) : (
        <p className="normal-zone-note">
          그래프는 업로드된 CSV의 센서 패턴 흐름을 참고용으로 표시합니다. 최종
          판단은 {getDisplayStatusText(result.status)}이며, Fall Alert가
          아니므로 DB에는 저장하지 않습니다.
        </p>
      )}
    </div>
  );
}

function DecisionEvidenceChart({ result }) {
  const fallProb = getDisplayFallProbability(result);
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
      label: "낙상 위험도",
      desc: "AI 모델이 낙상으로 판단한 정도",
      value: fallProb,
      max: 1,
      displayValue: result ? `${Math.round(fallProb * 100)}%` : "-",
      level: getLevel("fallProb", fallProb),
    },
    {
      key: "heightDrop",
      label: "높이 변화",
      desc: "대상의 높이 변화 정도",
      value: heightDrop,
      max: 1.5,
      displayValue: result ? `${heightDrop.toFixed(4)} m` : "-",
      level: getLevel("heightDrop", heightDrop),
    },
    {
      key: "speedMax",
      label: "최대 속도",
      desc: "순간적으로 움직인 최대 속도",
      value: speedMax,
      max: 2,
      displayValue: result ? `${speedMax.toFixed(4)} m/s` : "-",
      level: getLevel("speedMax", speedMax),
    },
    {
      key: "movementAfter",
      label: "이후 이동거리",
      desc: "동작 이후 대상자가 움직인 거리",
      value: movementAfter,
      max: 1,
      displayValue: result ? `${(movementAfter * 100).toFixed(2)} cm` : "-",
      level: getLevel("movementAfter", movementAfter),
    },
  ];

  const getDecisionReason = () => {
    if (!result) {
      return {
        title: "아직 예측 전입니다.",
        reasons: ["CSV 파일을 업로드하면 판단 근거가 표시됩니다."],
      };
    }

    const reasons = [];
    const isFallAlert = result.status === "Fall Alert";
    const movementCm = movementAfter * 100;

    if (isFallAlert) {
      if (fallProb >= 0.7) {
        reasons.push("AI 모델의 낙상 위험도가 높게 나왔습니다.");
      }

      if (heightDrop >= 0.8) {
        reasons.push("높이 변화가 크게 나타나 낙상 패턴과 유사합니다.");
      } else if (heightDrop >= 0.4) {
        reasons.push("높이 변화가 일부 감지되었습니다.");
      }

      if (speedMax >= 1.2) {
        reasons.push("순간 속도가 높아 갑작스러운 움직임이 감지되었습니다.");
      } else if (speedMax >= 0.6) {
        reasons.push("움직임 속도 변화가 감지되었습니다.");
      }

      if (movementAfter <= 0.2) {
        reasons.push(
          `동작 이후 이동거리가 ${movementCm.toFixed(
            2,
          )}cm로 낮아 움직임이 거의 없는 패턴이 나타났습니다.`,
        );
      }

      if (reasons.length === 0) {
        reasons.push(
          "전체 센서 패턴이 모델 기준에서 낙상 알림으로 분류되었습니다.",
        );
      }

      return {
        title: "낙상으로 판단한 이유",
        reasons,
      };
    }

    if (fallProb <= 0.01) {
      reasons.push("AI 모델의 낙상 위험도가 0%로 낮게 나왔습니다.");
    } else if (fallProb < 0.3) {
      reasons.push("AI 모델의 낙상 위험도가 낮은 수준입니다.");
    } else {
      reasons.push("AI 모델의 낙상 위험도가 알림 기준에 도달하지 않았습니다.");
    }

    if (heightDrop >= 0.4) {
      reasons.push(
        "높이 변화가 나타났지만, 모델은 이를 낙상이 아닌 정상적인 자세 변화로 판단했습니다.",
      );
    } else {
      reasons.push("높이 변화가 크지 않아 낙상 가능성이 낮습니다.");
    }

    if (speedMax >= 0.6) {
      reasons.push(
        "속도 변화가 일부 있었지만, 전체 패턴이 낙상 알림 기준에는 도달하지 않았습니다.",
      );
    } else {
      reasons.push("움직임 속도가 크지 않았습니다.");
    }

    if (movementAfter <= 0.2) {
      reasons.push(
        `동작 이후 이동거리는 ${movementCm.toFixed(
          2,
        )}cm로 작게 나타났지만, 최종 모델 판단은 정상 행동입니다.`,
      );
    } else {
      reasons.push(
        `동작 이후 이동거리가 ${movementCm.toFixed(2)}cm로 확인되었습니다.`,
      );
    }

    return {
      title: "정상으로 판단한 이유",
      reasons: [
        ...reasons,
        "최종 판단이 정상 행동이므로 Fall Alert로 처리하지 않았고 DB에도 저장하지 않았습니다.",
      ],
    };
  };

  const decision = getDecisionReason();

  return (
    <div className="chart-card evidence-card">
      <h3>낙상 판단 근거 그래프</h3>
      <p>
        낙상 위험도, 높이 변화, 최대 속도, 이후 이동거리를 기준으로 모델이 왜
        낙상/정상으로 판단했는지 보여줍니다.
      </p>

      <div className="bar-chart evidence-list">
        {rows.map((item) => {
          const width = result
            ? Math.max(
                4,
                Math.min(100, (Number(item.value || 0) / item.max) * 100),
              )
            : 4;

          return (
            <div className="evidence-item" key={item.key}>
              <div className="evidence-item-top">
                <div className="bar-label">
                  <strong>{item.label}</strong>
                  <span>{item.desc}</span>
                </div>

                <div className="bar-value" title={item.displayValue}>
                  {item.displayValue}
                </div>
              </div>

              <div className="bar-track">
                <div
                  className={`bar-fill evidence-${item.level}`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="evidence-summary">
        <h4>{decision.title}</h4>

        {result && (
          <p className="decision-status">
            최종 판단: <strong>{getDisplayStatusText(result.status)}</strong>
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
  if (!events || events.length === 0) {
    return (
      <div className="chart-card history-chart-card">
        <h3>최근 낙상 알림 위험도 그래프</h3>
        <p className="empty-chart">저장된 낙상 알림 데이터가 없습니다.</p>
      </div>
    );
  }

  return (
    <div className="chart-card history-chart-card">
      <h3>최근 낙상 알림 위험도 그래프</h3>
      <p>현재 페이지에 표시된 Fall Alert 이벤트의 낙상 위험도입니다.</p>

      <div className="mini-column-chart compact-history-bars">
        {events.map((event) => {
          const prob = normalizeProbability(
            event.raw_model_fall_prob ?? event.fall_prob,
          );
          const percent = Math.round(prob * 100);
          const height = Math.max(5, Math.min(100, percent));

          return (
            <div className="mini-column-item" key={getEventId(event)}>
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

/* =========================================================
   메인 컴포넌트
========================================================= */

function FallDashboard() {
  const [serverInfo, setServerInfo] = React.useState(null);
  const [stats, setStats] = React.useState(null);
  const [events, setEvents] = React.useState([]);
  const [eventPage, setEventPage] = React.useState(1);
  const [selectedEventId, setSelectedEventId] = React.useState(null);

  const [selectedFile, setSelectedFile] = React.useState(null);
  const [predictResult, setPredictResult] = React.useState(null);
  const [fallScoreData, setFallScoreData] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [message, setMessage] = React.useState("");

  const totalPages = Math.max(1, Math.ceil(events.length / EVENT_PAGE_SIZE));
  const safeEventPage = Math.min(eventPage, totalPages);
  const pagedEvents = events.slice(
    (safeEventPage - 1) * EVENT_PAGE_SIZE,
    safeEventPage * EVENT_PAGE_SIZE,
  );

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
      const res = await apiGet("/events?limit=50");
      const list = res.data.events || [];
      setEvents(list);
      setEventPage(1);
    } catch (err) {
      setEvents([]);
      setEventPage(1);
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

  React.useEffect(() => {
    if (eventPage > totalPages) {
      setEventPage(totalPages);
    }
  }, [eventPage, totalPages]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];

    setSelectedFile(file);
    setPredictResult(null);
    setMessage("");
    setFallScoreData([]);
    setSelectedEventId(null);

    if (!file) return;

    const reader = new FileReader();

    reader.onload = (event) => {
      const text = event.target.result;
      const parsed = parseFallScoreFromCsv(text);

      setFallScoreData(parsed.rows);
    };

    reader.onerror = () => {
      setFallScoreData([]);
    };

    reader.readAsText(file);
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
      setSelectedEventId(null);

      if (res.data.status === "Fall Alert") {
        setMessage("낙상 알림이 감지되어 저장했습니다.");
      } else {
        setMessage("정상 행동으로 판단되었습니다. DB에는 저장하지 않습니다.");
      }

      await loadStats();
      await loadEvents();
    } catch (err) {
      setMessage("예측 요청 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectEvent = (event) => {
    const eventId = getEventId(event);

    const historyResult = {
      ...event,
      file_name: event.file_name || "-",
      status: event.status || "Fall Alert",
      alert:
        event.alert !== undefined ? event.alert : event.status === "Fall Alert",
      fall_prob: event.fall_prob ?? event.raw_model_fall_prob ?? 0,
      raw_model_fall_prob:
        event.raw_model_fall_prob ??
        event.raw_fall_prob ??
        event.fall_prob ??
        0,
      speed_max: event.speed_max ?? 0,
      height_drop: event.height_drop ?? 0,
      movement_after: event.movement_after ?? 0,
      db_saved: true,
      db_message: "저장된 과거 낙상 알림 이벤트입니다.",
    };

    setSelectedFile(null);
    setFallScoreData([]);
    setPredictResult(historyResult);
    setSelectedEventId(eventId);
    setMessage("과거 낙상 알림 로그를 불러왔습니다.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleDeleteEvents = async () => {
    const ok = window.confirm("저장된 낙상 알림 로그를 모두 삭제할까요?");
    if (!ok) return;

    try {
      await apiDelete("/events");
      await loadStats();
      await loadEvents();
      setPredictResult(null);
      setFallScoreData([]);
      setSelectedEventId(null);
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
            저장 정책: <strong>Fall Alert만 저장</strong>, 정상 행동은 저장하지
            않음
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

              <p className="result-message">
                {getDisplayResultMessage(predictResult)}
              </p>

              <div className="result-grid">
                <div>
                  <span>파일명</span>
                  <strong title={predictResult.file_name}>
                    {predictResult.file_name}
                  </strong>
                </div>
                <div>
                  <span>낙상 위험도</span>
                  <strong>{getDisplayFallPercent(predictResult)}%</strong>
                </div>
                <div>
                  <span>속도 최댓값</span>
                  <strong>{formatSpeed(predictResult.speed_max)}</strong>
                </div>
                <div>
                  <span>높이 변화</span>
                  <strong>{formatMeter(predictResult.height_drop)}</strong>
                </div>
                <div>
                  <span>낙상 이후 이동거리</span>
                  <strong>
                    {formatMovementCm(predictResult.movement_after)}
                  </strong>
                </div>
                <div>
                  <span>DB 저장</span>
                  <strong>
                    {predictResult.db_saved ? "저장됨" : "저장 안 함"}
                  </strong>
                </div>
              </div>

              {predictResult.db_message && (
                <p className="db-message">
                  {predictResult.status === "Fall Alert"
                    ? predictResult.db_message
                    : "정상 행동은 MongoDB에 저장하지 않습니다."}
                </p>
              )}
            </div>
          )}

          {predictResult && (
            <div className="analysis-layout">
              <div className="analysis-left">
                <ProbabilityChart result={predictResult} />
                <FallScoreChart data={fallScoreData} result={predictResult} />
              </div>

              <div className="analysis-right">
                <DecisionEvidenceChart result={predictResult} />
              </div>
            </div>
          )}
        </section>

        <section className="card events-card">
          <div className="section-title-row">
            <div>
              <h2>최근 낙상 알림 로그</h2>
              <p className="desc">
                저장된 Fall Alert 이벤트만 표시합니다. 표의 행을 클릭하면 과거
                결과값을 위쪽 결과 영역에서 다시 볼 수 있습니다.
              </p>
            </div>

            <button className="delete-btn" onClick={handleDeleteEvents}>
              로그 삭제
            </button>
          </div>

          <RecentEventChart events={pagedEvents} />

          <div className="table-wrap history-table-wrap">
            <table className="history-log-table">
              <thead>
                <tr>
                  <th>시간</th>
                  <th>상태</th>
                  <th>알림</th>
                  <th>파일명</th>
                  <th>위험도</th>
                  <th>ID</th>
                </tr>
              </thead>

              <tbody>
                {pagedEvents.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="empty">
                      저장된 낙상 알림이 없습니다.
                    </td>
                  </tr>
                ) : (
                  pagedEvents.map((event) => {
                    const eventPercent = Math.round(
                      normalizeProbability(
                        event.raw_model_fall_prob ?? event.fall_prob,
                      ) * 100,
                    );
                    const eventId = getEventId(event);
                    const isSelected = selectedEventId === eventId;

                    return (
                      <tr
                        key={eventId}
                        className={`clickable-row ${
                          isSelected ? "selected-row" : ""
                        }`}
                        onClick={() => handleSelectEvent(event)}
                        title="클릭하면 과거 결과값을 불러옵니다."
                      >
                        <td className="time-cell">
                          {formatDateTime(event.created_at)}
                        </td>
                        <td className="status-cell">
                          <StatusBadge status={event.status} />
                        </td>
                        <td className="alert-cell">
                          <AlertText alert={event.alert} />
                        </td>
                        <td className="file-cell" title={event.file_name}>
                          {event.file_name}
                        </td>
                        <td className="prob-cell">{eventPercent}%</td>
                        <td className="id-cell">{String(eventId).slice(-6)}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <button
              type="button"
              onClick={() => setEventPage((page) => Math.max(1, page - 1))}
              disabled={safeEventPage <= 1}
            >
              &lt;
            </button>

            <span>
              {safeEventPage} / {totalPages}
            </span>

            <button
              type="button"
              onClick={() =>
                setEventPage((page) => Math.min(totalPages, page + 1))
              }
              disabled={safeEventPage >= totalPages}
            >
              &gt;
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default FallDashboard;

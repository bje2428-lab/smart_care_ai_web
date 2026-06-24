import React from "react";
import axios from "axios";
import "./FallDashboard.css";

const API_URLS = (
  import.meta.env.VITE_API_URLS ||
  import.meta.env.VITE_API_URL ||
  ""
)
  .split(",")
  .map((url) => url.trim().replace(/\/+$/, ""))
  .filter(Boolean);

const FPS = 20;
const FRAME_WINDOW_SIZE = 10;
const REALTIME_INTERVAL_MS = 650;
const EVENT_PAGE_SIZE = 5;
const LIVE_PAGE_SIZE = 5;

let activeApiBaseUrl = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getErrorMessage(error) {
  if (error?.code === "ECONNABORTED") return "요청 시간 초과";
  if (error?.response?.status) return `HTTP ${error.response.status}`;
  if (error?.message) return error.message;
  return "알 수 없는 오류";
}

async function requestApi(path, options = {}) {
  if (API_URLS.length === 0) {
    throw new Error(".env에 VITE_API_URLS 또는 VITE_API_URL이 없습니다.");
  }

  const urls = activeApiBaseUrl
    ? [activeApiBaseUrl, ...API_URLS.filter((url) => url !== activeApiBaseUrl)]
    : API_URLS;

  const errors = [];

  for (const baseUrl of urls) {
    try {
      const res = await axios({
        method: options.method || "get",
        url: `${baseUrl}${path}`,
        data: options.data,
        headers: options.headers,
        timeout: options.timeout || 10000,
      });

      activeApiBaseUrl = baseUrl;
      return res;
    } catch (error) {
      errors.push(`${baseUrl}: ${getErrorMessage(error)}`);
      console.log(`${baseUrl} 연결 실패`, error);
    }
  }

  throw new Error(`백엔드 연결 실패: ${errors.join(" / ")}`);
}

async function apiGet(path) {
  return requestApi(path, {
    method: "get",
    timeout: 10000,
  });
}

async function apiPostForm(path, formData, saveEvent = true) {
  const query = path.includes("?")
    ? `&save_event=${saveEvent}`
    : `?save_event=${saveEvent}`;

  return requestApi(`${path}${query}`, {
    method: "post",
    data: formData,
    timeout: 60000,
  });
}

async function apiDelete(path) {
  return requestApi(path, {
    method: "delete",
    timeout: 10000,
  });
}

function toSafeNumber(value) {
  if (value === null || value === undefined || value === "") return 0;

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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getFromResult(result, keys) {
  if (!result) return undefined;

  const sources = [
    result,
    result.features,
    result.metrics,
    result.sensor,
    result.result,
    result.data,
  ];

  for (const source of sources) {
    if (!source) continue;

    for (const key of keys) {
      if (source[key] !== undefined && source[key] !== null) {
        return source[key];
      }
    }
  }

  return undefined;
}

function getFirstNumber(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") {
      return toSafeNumber(value);
    }
  }

  return 0;
}

function normalizeFallResult(data, fallbackFileName = "-") {
  const result = data || {};

  const rawProb = getFromResult(result, [
    "raw_model_fall_prob",
    "fall_prob",
    "fall_probability",
    "fall_risk",
    "fall_risk_percent",
    "risk_score",
    "probability",
  ]);

  const fallProb = normalizeProbability(rawProb);

  const speedMax = getFirstNumber(
    getFromResult(result, [
      "speed_max",
      "max_speed",
      "v_abs_max",
      "abs_v_max",
      "v_max_abs",
      "v_max",
      "velocity_max",
    ]),
  );

  const heightDrop = getFirstNumber(
    getFromResult(result, [
      "height_drop",
      "z_drop",
      "z_range",
      "z_center_drop",
      "z_center_first_to_min_drop",
      "z_center_peak_to_last_drop",
      "max_height_drop",
    ]),
  );

  const movementAfter = getFirstNumber(
    getFromResult(result, [
      "movement_after",
      "movement_after_fall",
      "after_movement",
      "tail_movement",
      "movement_after_mean",
      "center_move_tail_mean",
      "center_move_after",
    ]),
  );

  let status =
    result.status ||
    result.prediction ||
    result.label ||
    result.result_label ||
    "Normal";

  if (status === "Exception") status = "Normal";

  if (fallProb >= 0.7) {
    status = "Fall Alert";
  }

  const alert =
    result.alert !== undefined
      ? Boolean(result.alert)
      : status === "Fall Alert";

  const windowResults = Array.isArray(result.window_results)
    ? result.window_results.map((item, index) =>
        normalizeFallResult(
          {
            ...item,
            chunk_index: item.chunk_index ?? index,
            file_name: result.file_name || fallbackFileName,
          },
          result.file_name || fallbackFileName,
        ),
      )
    : [];

  return {
    ...result,
    status,
    alert,

    file_name:
      result.file_name ||
      result.filename ||
      result.file ||
      fallbackFileName ||
      "-",
    filename:
      result.filename ||
      result.file_name ||
      result.file ||
      fallbackFileName ||
      "-",

    raw_model_fall_prob: fallProb,
    fall_prob: fallProb,
    fall_probability: fallProb,

    speed_max: speedMax,
    max_speed: speedMax,
    height_drop: heightDrop,
    movement_after: movementAfter,

    frame_start:
      result.frame_start !== undefined && result.frame_start !== null
        ? toSafeNumber(result.frame_start)
        : null,
    frame_end:
      result.frame_end !== undefined && result.frame_end !== null
        ? toSafeNumber(result.frame_end)
        : null,

    frame_count: toSafeNumber(result.frame_count),
    point_count: toSafeNumber(result.point_count),

    window_results: windowResults,

    action_case: result.action_case || "unknown",
    action_label: result.action_label || "행동 케이스 분석 없음",
    behavior_pattern:
      result.behavior_pattern || "행동 패턴 분석 정보가 없습니다.",
    likely_cause: result.likely_cause || "추정 원인 정보가 없습니다.",
    analysis_summary: result.analysis_summary || "분석 요약 정보가 없습니다.",
    direction: result.direction || "-",
    risk_reason: Array.isArray(result.risk_reason) ? result.risk_reason : [],
    recommendations: Array.isArray(result.recommendations)
      ? result.recommendations
      : [],

    db_saved: Boolean(result.db_saved),

    message:
      result.message ||
      (status === "Fall Alert"
        ? "낙상 알림으로 판단되었습니다."
        : "정상 행동으로 판단되었습니다. 낙상 알림 기준에 도달하지 않아 DB에는 저장하지 않습니다."),
  };
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

function getDisplayFallPercent(result) {
  return Math.round(getModelFallProbability(result) * 100);
}

function getRiskLevelText(percent) {
  if (percent >= 85) return "매우 높음";
  if (percent >= 70) return "위험";
  if (percent >= 40) return "주의";
  return "낮음";
}

function getDisplayStatus(status) {
  if (status === "Exception") return "Normal";
  return status || "-";
}

function formatDateTime(value) {
  if (!value) return "-";
  return String(value).replace("T", " ").replace(".000Z", "").slice(0, 19);
}

function formatSpeed(value) {
  return `${toSafeNumber(value).toFixed(4)} m/s`;
}

function formatMeter(value) {
  return `${toSafeNumber(value).toFixed(4)} m`;
}

function formatMovementCm(value) {
  return `${(toSafeNumber(value) * 100).toFixed(2)} cm`;
}

function formatFrameRange(result) {
  if (!result) return "-";

  const hasStart =
    result.frame_start !== null && result.frame_start !== undefined;
  const hasEnd = result.frame_end !== null && result.frame_end !== undefined;

  if (!hasStart && !hasEnd) return "-";

  return `${result.frame_start ?? "-"} ~ ${result.frame_end ?? "-"}`;
}

function getEventId(event) {
  return String(
    event?._id ||
      event?.event_id ||
      event?.id ||
      `${event?.created_at || ""}-${event?.file_name || ""}`,
  );
}

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

function parseCsvText(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return {
      headers: [],
      dataLines: [],
      frameIndex: -1,
      xIndex: -1,
      yIndex: -1,
      zIndex: -1,
      vIndex: -1,
    };
  }

  const headers = parseCsvLine(lines[0]).map((h) => h.trim());
  const lower = headers.map((h) => h.toLowerCase());

  return {
    headers,
    dataLines: lines.slice(1),
    frameIndex: lower.indexOf("frame"),
    xIndex: lower.indexOf("x"),
    yIndex: lower.indexOf("y"),
    zIndex: lower.indexOf("z"),
    vIndex: lower.indexOf("v"),
  };
}

function getFileBaseName(fileName) {
  return String(fileName || "uploaded")
    .replace(/\.csv$/i, "")
    .replace(/[^\w가-힣.-]/g, "_");
}

function buildFrameChunksFromCsv(text, originalFileName) {
  const parsed = parseCsvText(text);

  if (
    parsed.headers.length === 0 ||
    parsed.frameIndex === -1 ||
    parsed.xIndex === -1 ||
    parsed.yIndex === -1 ||
    parsed.zIndex === -1 ||
    parsed.vIndex === -1
  ) {
    return {
      chunks: [],
      error: "CSV에 frame, x, y, z, v 컬럼이 필요합니다.",
    };
  }

  const rows = parsed.dataLines
    .map((line) => {
      const cols = parseCsvLine(line);
      const frame = Number(cols[parsed.frameIndex]);

      return {
        line,
        frame,
        x: toSafeNumber(cols[parsed.xIndex]),
        y: toSafeNumber(cols[parsed.yIndex]),
        z: toSafeNumber(cols[parsed.zIndex]),
        v: toSafeNumber(cols[parsed.vIndex]),
      };
    })
    .filter((row) => Number.isFinite(row.frame));

  const uniqueFrames = [...new Set(rows.map((row) => row.frame))].sort(
    (a, b) => a - b,
  );

  if (uniqueFrames.length === 0) {
    return {
      chunks: [],
      error: "frame 값이 올바르지 않습니다.",
    };
  }

  const chunks = [];
  const baseName = getFileBaseName(originalFileName);

  for (let i = 0; i < uniqueFrames.length; i += FRAME_WINDOW_SIZE) {
    const frameGroup = uniqueFrames.slice(i, i + FRAME_WINDOW_SIZE);
    const frameSet = new Set(frameGroup);

    const chunkLines = rows
      .filter((row) => frameSet.has(row.frame))
      .map((row) => row.line);

    if (chunkLines.length === 0) continue;

    const startFrame = frameGroup[0];
    const endFrame = frameGroup[frameGroup.length - 1];

    chunks.push({
      index: chunks.length,
      startFrame,
      endFrame,
      frameCount: frameGroup.length,
      rowCount: chunkLines.length,
      secondStart: startFrame / FPS,
      secondEnd: (endFrame + 1) / FPS,
      fileName: `${baseName}_frame_${startFrame}_${endFrame}.csv`,
      csvText: `${parsed.headers.join(",")}\n${chunkLines.join("\n")}\n`,
    });
  }

  return {
    chunks,
    error: "",
  };
}

function StatusBadge({ status }) {
  const displayStatus = getDisplayStatus(status);

  const className =
    displayStatus === "Fall Alert"
      ? "badge danger"
      : displayStatus === "Normal"
        ? "badge normal"
        : "badge";

  return <span className={className}>{displayStatus}</span>;
}

function AlertText({ alert }) {
  return alert ? (
    <span className="alert-true">알림 발생</span>
  ) : (
    <span className="alert-false">알림 없음</span>
  );
}

function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div className="result-box">
      <div className="result-header">
        <StatusBadge status={result.status} />
        <AlertText alert={result.alert} />
      </div>

      <p className="result-message">{result.message}</p>

      <div className="result-grid">
        <div>
          <span>파일명</span>
          <strong title={result.file_name}>{result.file_name}</strong>
        </div>

        <div>
          <span>가장 위험한 프레임</span>
          <strong>{formatFrameRange(result)}</strong>
        </div>

        <div>
          <span>최종 낙상 위험도</span>
          <strong>{getDisplayFallPercent(result)}%</strong>
        </div>

        <div>
          <span>최대 속도</span>
          <strong>{formatSpeed(result.speed_max)}</strong>
        </div>

        <div>
          <span>높이 변화</span>
          <strong>{formatMeter(result.height_drop)}</strong>
        </div>

        <div>
          <span>이후 이동거리</span>
          <strong>{formatMovementCm(result.movement_after)}</strong>
        </div>

        <div>
          <span>분석 구간 수</span>
          <strong>
            {result.window_count || result.window_results?.length || "-"}
          </strong>
        </div>

        <div>
          <span>DB 저장</span>
          <strong>{result.db_saved ? "저장됨" : "저장 안 함"}</strong>
        </div>
      </div>

      {result.db_message && <p className="db-message">{result.db_message}</p>}
    </div>
  );
}

function BehaviorAnalysisCard({ result }) {
  if (!result) return null;

  const isFallAlert = result.status === "Fall Alert";

  return (
    <section className="card behavior-full-card">
      <div className="chart-title-row">
        <div>
          <h2>
            {isFallAlert
              ? "낙상 행동 패턴 및 원인 분석"
              : "정상 행동 패턴 분석"}
          </h2>
          <p className="desc">
            {isFallAlert
              ? "낙상 위험 구간, 속도, 높이 변화, 이후 움직임을 기반으로 원인을 추정합니다."
              : "현재 데이터가 낙상 기준에 도달하지 않은 이유를 분석합니다."}
          </p>
        </div>

        <span className="behavior-chip">{result.action_label}</span>
      </div>

      <div className="behavior-summary">
        <strong>{result.analysis_summary}</strong>
        <p>{result.behavior_pattern}</p>
      </div>

      <div className="behavior-grid">
        <div>
          <span>행동 케이스</span>
          <strong>{result.action_label}</strong>
        </div>

        <div>
          <span>추정 방향</span>
          <strong>{result.direction || "-"}</strong>
        </div>

        <div>
          <span>{isFallAlert ? "추정 원인" : "정상 판단 이유"}</span>
          <strong>{result.likely_cause}</strong>
        </div>
      </div>

      <div className="behavior-list-grid">
        <div>
          <h4>판단 근거</h4>
          <ul>
            {(result.risk_reason || []).length === 0 ? (
              <li>아직 판단 근거가 없습니다.</li>
            ) : (
              result.risk_reason.map((item, index) => (
                <li key={`reason-${index}`}>{item}</li>
              ))
            )}
          </ul>
        </div>

        <div>
          <h4>대응 권장</h4>
          <ul>
            {(result.recommendations || []).length === 0 ? (
              <li>추가 대응 권장 내용이 없습니다.</li>
            ) : (
              result.recommendations.map((item, index) => (
                <li key={`recommend-${index}`}>{item}</li>
              ))
            )}
          </ul>
        </div>
      </div>
    </section>
  );
}

function ProbabilityChart({ result }) {
  if (!result) return null;

  const percent = Math.max(0, Math.min(100, getDisplayFallPercent(result)));

  return (
    <div className="chart-card">
      <div className="chart-title-row">
        <div>
          <h3>낙상 위험도</h3>
          <p>전체 파일과 10프레임 분석 결과를 같은 기준으로 표시합니다.</p>
        </div>
        <StatusBadge status={result.status} />
      </div>

      <div className="probability-graph">
        <div className="probability-track">
          <div
            className={`probability-fill ${
              percent >= 70
                ? "danger-fill"
                : percent >= 40
                  ? "warning-fill"
                  : "normal-fill"
            }`}
            style={{ width: `${percent}%` }}
          />
        </div>

        <div className="probability-info">
          <strong>{percent}%</strong>
          <span>낙상 위험도: {getRiskLevelText(percent)}</span>
        </div>
      </div>

      <div className="scale-row">
        <span>0%</span>
        <span>40%</span>
        <span>70%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function getTimelineRows(result, liveResults) {
  if (liveResults && liveResults.length > 0) {
    return [...liveResults].reverse().map((item, index) => ({
      index,
      frame: item.frame_end ?? index,
      frame_start: item.frame_start,
      frame_end: item.frame_end,
      displayScore: getDisplayFallPercent(item),
      status: item.status,
    }));
  }

  if (result?.window_results?.length > 0) {
    return result.window_results.map((item, index) => ({
      index,
      frame: item.frame_end ?? index,
      frame_start: item.frame_start,
      frame_end: item.frame_end,
      displayScore: getDisplayFallPercent(item),
      status: item.status,
    }));
  }

  return [];
}

function getXByIndex(index, dataLength, width, padding) {
  const usableWidth = width - padding.left - padding.right;
  return padding.left + (index / Math.max(1, dataLength - 1)) * usableWidth;
}

function getYByScore(score, height, padding) {
  const usableHeight = height - padding.top - padding.bottom;
  return padding.top + (1 - clamp(score, 0, 100) / 100) * usableHeight;
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
    .map((item, index) => ({ item, index }))
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
    startFrame: displayData[left].frame_start,
    centerFrame: best.item.frame,
    endFrame: displayData[right].frame_end,
    peakScore: Math.round(best.item.displayScore),
    startIndex: left,
    centerIndex: best.index,
    endIndex: right,
  };
}

function FallScoreChart({ result, liveResults }) {
  if (!result && (!liveResults || liveResults.length === 0)) return null;

  const data = getTimelineRows(result, liveResults);

  if (!data || data.length === 0) {
    return (
      <div className="chart-card">
        <h3>프레임별 낙상 판정 그래프</h3>
        <p className="empty-chart">아직 분석된 10프레임 구간이 없습니다.</p>
      </div>
    );
  }

  const threshold = 70;
  const displayZone = detectDisplayFallZone(data, threshold);

  const width = 760;
  const height = 280;
  const padding = {
    top: 32,
    right: 30,
    bottom: 42,
    left: 54,
  };

  const firstFrame = data[0]?.frame_start ?? 0;
  const lastFrame = data[data.length - 1]?.frame_end ?? data.length;

  const showZone = Boolean(displayZone);
  const thresholdY = getYByScore(threshold, height, padding);

  const zoneX1 = showZone
    ? getXByIndex(displayZone.startIndex, data.length, width, padding)
    : 0;

  const zoneX2 = showZone
    ? getXByIndex(displayZone.endIndex, data.length, width, padding)
    : 0;

  const centerX = showZone
    ? getXByIndex(displayZone.centerIndex, data.length, width, padding)
    : 0;

  const maxDisplayScore = Math.round(
    Math.max(...data.map((item) => item.displayScore), 0),
  );

  return (
    <div className="chart-card">
      <div className="chart-title-row">
        <div>
          <h3>프레임별 낙상 판정 그래프</h3>
          <p>10프레임 분석 결과가 실시간으로 선형차트에 반영됩니다.</p>
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
          preserveAspectRatio="none"
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
            points={getScoreSvgPoints(data, width, height, padding)}
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

      <div className="fall-metric-grid">
        <div>
          <span>최종 낙상 위험도</span>
          <strong>
            {result ? getDisplayFallPercent(result) : maxDisplayScore}%
          </strong>
          <p>가장 위험한 10프레임 기준</p>
        </div>

        <div>
          <span>그래프 최고점</span>
          <strong>{maxDisplayScore}%</strong>
          <p>{maxDisplayScore >= threshold ? "기준선 초과" : "기준선 미만"}</p>
        </div>

        <div>
          <span>분석 단위</span>
          <strong>{FRAME_WINDOW_SIZE}프레임</strong>
          <p>약 {((FRAME_WINDOW_SIZE / FPS) * 2).toFixed(1)}초 단위</p>
        </div>

        <div>
          <span>분석 구간</span>
          <strong>{data.length}개</strong>
          <p>실시간 구간 누적</p>
        </div>
      </div>

      {showZone ? (
        <p className="fall-zone-note">
          낙상 의심 구간: frame {displayZone.startFrame} ~{" "}
          {displayZone.endFrame}
        </p>
      ) : (
        <p className="normal-zone-note">
          낙상 기준선을 넘은 구간이 없어 정상 흐름으로 표시됩니다.
        </p>
      )}
    </div>
  );
}

function EvidenceChart({ result }) {
  if (!result) return null;

  const fallProb = getModelFallProbability(result);
  const heightDrop = toSafeNumber(result.height_drop);
  const speedMax = toSafeNumber(result.speed_max);
  const movementAfter = toSafeNumber(result.movement_after);

  const rows = [
    {
      key: "fallProb",
      label: "낙상 위험도",
      desc: "가장 위험한 10프레임 구간 기준",
      value: fallProb,
      max: 1,
      displayValue: `${Math.round(fallProb * 100)}%`,
    },
    {
      key: "heightDrop",
      label: "높이 변화",
      desc: "대상의 높이 변화 정도",
      value: heightDrop,
      max: 1.5,
      displayValue: `${heightDrop.toFixed(4)} m`,
    },
    {
      key: "speedMax",
      label: "최대 속도",
      desc: "순간적으로 움직인 최대 속도",
      value: speedMax,
      max: 2,
      displayValue: `${speedMax.toFixed(4)} m/s`,
    },
    {
      key: "movementAfter",
      label: "이후 이동거리",
      desc: "동작 이후 대상자가 움직인 거리",
      value: movementAfter,
      max: 1,
      displayValue: `${(movementAfter * 100).toFixed(2)} cm`,
    },
  ];

  const isFallAlert = result.status === "Fall Alert";

  return (
    <div className="chart-card evidence-card">
      <h3>낙상 판단 근거 그래프</h3>
      <p>위험도, 높이 변화, 최대 속도, 이후 이동거리를 보여줍니다.</p>

      <div className="bar-chart evidence-list">
        {rows.map((item) => {
          const width = Math.max(
            4,
            Math.min(100, (item.value / item.max) * 100),
          );

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
                  className={`bar-fill ${
                    width >= 70
                      ? "evidence-danger"
                      : width >= 40
                        ? "evidence-warning"
                        : "evidence-normal"
                  }`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="evidence-summary">
        <h4>{isFallAlert ? "낙상으로 판단한 이유" : "정상으로 판단한 이유"}</h4>
        <p className="decision-status">
          최종 판단: <strong>{getDisplayStatus(result.status)}</strong>
        </p>
        <ul>
          {isFallAlert ? (
            <>
              <li>10프레임 구간 중 낙상 기준을 넘는 패턴이 감지되었습니다.</li>
              <li>
                전체 파일과 실시간 분석은 같은 최고 위험 구간 기준으로
                표시됩니다.
              </li>
            </>
          ) : (
            <>
              <li>
                모든 10프레임 구간이 Fall Alert 기준에 도달하지 않았습니다.
              </li>
              <li>정상 행동으로 처리되어 DB에는 저장하지 않습니다.</li>
            </>
          )}
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
          const normalized = normalizeFallResult(event, event.file_name || "-");
          const percent = getDisplayFallPercent(normalized);
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

function RealtimePanel({
  selectedFile,
  frameChunks,
  realtimeRunning,
  currentChunkIndex,
  liveResults,
  livePage,
  setLivePage,
}) {
  const currentChunk =
    currentChunkIndex >= 0 ? frameChunks[currentChunkIndex] : null;

  const totalPages = Math.max(
    1,
    Math.ceil(liveResults.length / LIVE_PAGE_SIZE),
  );
  const safePage = Math.min(Math.max(1, livePage), totalPages);

  const startIndex = (safePage - 1) * LIVE_PAGE_SIZE;
  const endIndex = startIndex + LIVE_PAGE_SIZE;
  const visibleRows = liveResults.slice(startIndex, endIndex);

  React.useEffect(() => {
    if (livePage > totalPages) {
      setLivePage(totalPages);
    }
  }, [livePage, totalPages, setLivePage]);

  return (
    <div className="realtime-panel compact-realtime-panel">
      <div className="realtime-title-row">
        <div>
          <h3>10프레임 실시간 시뮬레이션</h3>
          <p>
            전체 구간은 모두 보관하고, 표는 5개씩 페이지로 나누어 표시합니다.
          </p>
        </div>

        <span className={realtimeRunning ? "live-chip on" : "live-chip"}>
          {realtimeRunning ? "실시간 실행 중" : "대기 중"}
        </span>
      </div>

      <div className="realtime-grid compact-realtime-grid">
        <div>
          <span>선택 파일</span>
          <strong>{selectedFile?.name || "-"}</strong>
        </div>

        <div>
          <span>분석 단위</span>
          <strong>{FRAME_WINDOW_SIZE}프레임</strong>
        </div>

        <div>
          <span>전체 구간</span>
          <strong>{frameChunks.length}</strong>
        </div>

        <div>
          <span>현재 구간</span>
          <strong>
            {currentChunk
              ? `${currentChunk.index + 1} / ${frameChunks.length}`
              : "-"}
          </strong>
        </div>
      </div>

      <div className="compact-live-summary">
        <span>
          현재 프레임:{" "}
          <strong>
            {currentChunk
              ? `${currentChunk.startFrame} ~ ${currentChunk.endFrame}`
              : "-"}
          </strong>
        </span>

        <span>
          누적 분석: <strong>{liveResults.length}</strong>개 구간
        </span>

        <span>
          현재 페이지:{" "}
          <strong>
            {safePage} / {totalPages}
          </strong>
        </span>
      </div>

      <div className="table-wrap realtime-table-wrap compact-live-table-wrap">
        <table className="history-log-table compact-live-table">
          <thead>
            <tr>
              <th>구간</th>
              <th>프레임</th>
              <th>상태</th>
              <th>위험도</th>
              <th>속도</th>
              <th>높이</th>
            </tr>
          </thead>

          <tbody>
            {visibleRows.length === 0 ? (
              <tr>
                <td colSpan="6" className="empty">
                  아직 실시간 분석 결과가 없습니다.
                </td>
              </tr>
            ) : (
              visibleRows.map((item) => (
                <tr
                  key={`${item.chunk_index}-${item.frame_start}-${item.frame_end}`}
                >
                  <td>{item.chunk_index + 1}</td>

                  <td>
                    {item.frame_start} ~ {item.frame_end}
                  </td>

                  <td>
                    <StatusBadge status={item.status} />
                  </td>

                  <td>{getDisplayFallPercent(item)}%</td>

                  <td>{formatSpeed(item.speed_max)}</td>

                  <td>{formatMeter(item.height_drop)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="live-pagination">
        <button
          type="button"
          onClick={() => setLivePage((page) => Math.max(1, page - 1))}
          disabled={safePage <= 1}
        >
          &lt;
        </button>

        <span>
          {safePage} / {totalPages}
        </span>

        <button
          type="button"
          onClick={() => setLivePage((page) => Math.min(totalPages, page + 1))}
          disabled={safePage >= totalPages}
        >
          &gt;
        </button>
      </div>
    </div>
  );
}

function FallDashboard() {
  const [serverInfo, setServerInfo] = React.useState(null);
  const [fallInfo, setFallInfo] = React.useState(null);

  const [stats, setStats] = React.useState({
    total_events: 0,
    fall_alert_count: 0,
  });

  const [events, setEvents] = React.useState([]);
  const [eventPage, setEventPage] = React.useState(1);
  const [selectedEventId, setSelectedEventId] = React.useState(null);

  const [selectedFile, setSelectedFile] = React.useState(null);
  const [csvText, setCsvText] = React.useState("");
  const [frameChunks, setFrameChunks] = React.useState([]);

  const [predictResult, setPredictResult] = React.useState(null);
  const [message, setMessage] = React.useState("");

  const [loading, setLoading] = React.useState(false);
  const [realtimeRunning, setRealtimeRunning] = React.useState(false);
  const [currentChunkIndex, setCurrentChunkIndex] = React.useState(-1);
  const [liveResults, setLiveResults] = React.useState([]);
  const [livePage, setLivePage] = React.useState(1);

  const realtimeRunningRef = React.useRef(false);

  const totalPages = Math.max(1, Math.ceil(events.length / EVENT_PAGE_SIZE));
  const safeEventPage = Math.min(eventPage, totalPages);
  const pagedEvents = events.slice(
    (safeEventPage - 1) * EVENT_PAGE_SIZE,
    safeEventPage * EVENT_PAGE_SIZE,
  );

  const refreshAll = async () => {
    try {
      const healthRes = await apiGet("/health");
      setServerInfo(healthRes.data);
      setMessage("");

      try {
        const fallHealthRes = await apiGet("/fall/health");
        setFallInfo(fallHealthRes.data);
      } catch (err) {
        setFallInfo(null);
      }

      try {
        const statsRes = await apiGet("/stats");
        setStats(statsRes.data);
      } catch (err) {
        setStats({
          total_events: 0,
          fall_alert_count: 0,
        });
      }

      try {
        const eventsRes = await apiGet("/events?limit=50");
        const list = eventsRes.data.events || [];
        setEvents(list.map((item) => normalizeFallResult(item)));
        setEventPage(1);
      } catch (err) {
        setEvents([]);
      }
    } catch (err) {
      setServerInfo(null);
      setFallInfo(null);
      setStats({
        total_events: 0,
        fall_alert_count: 0,
      });
      setEvents([]);
      setMessage(
        `백엔드 연결 실패: FastAPI 서버를 먼저 켜야 합니다. (${getErrorMessage(err)})`,
      );
    }
  };

  React.useEffect(() => {
    refreshAll();

    return () => {
      realtimeRunningRef.current = false;
    };
  }, []);

  React.useEffect(() => {
    if (eventPage > totalPages) {
      setEventPage(totalPages);
    }
  }, [eventPage, totalPages]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];

    realtimeRunningRef.current = false;

    setSelectedFile(file || null);
    setCsvText("");
    setFrameChunks([]);
    setPredictResult(null);
    setMessage("");
    setLiveResults([]);
    setLivePage(1);
    setCurrentChunkIndex(-1);
    setRealtimeRunning(false);
    setSelectedEventId(null);

    if (!file) return;

    const reader = new FileReader();

    reader.onload = (event) => {
      const text = String(event.target.result || "");
      const chunkResult = buildFrameChunksFromCsv(text, file.name);

      setCsvText(text);

      if (chunkResult.error) {
        setFrameChunks([]);
        setMessage(chunkResult.error);
      } else {
        setFrameChunks(chunkResult.chunks);
        setMessage(
          `CSV를 10프레임 단위로 ${chunkResult.chunks.length}개 구간으로 나눴습니다.`,
        );
      }
    };

    reader.onerror = () => {
      setCsvText("");
      setFrameChunks([]);
      setMessage("CSV 파일을 읽는 중 오류가 발생했습니다.");
    };

    reader.readAsText(file);
  };

  const predictCsvChunk = async (chunk) => {
    const blob = new Blob([chunk.csvText], {
      type: "text/csv;charset=utf-8",
    });

    const chunkFile = new File([blob], chunk.fileName, {
      type: "text/csv",
    });

    const formData = new FormData();
    formData.append("file", chunkFile);

    const res = await apiPostForm("/predict", formData, false);
    const normalized = normalizeFallResult(res.data, chunk.fileName);

    return {
      ...normalized,
      chunk_index: chunk.index,
      frame_start: chunk.startFrame,
      frame_end: chunk.endFrame,
      frame_count: chunk.frameCount,
      row_count: chunk.rowCount,
      second_start: chunk.secondStart,
      second_end: chunk.secondEnd,
    };
  };

  const runWholeFileAggregatePredict = async (saveEvent = true) => {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await apiPostForm("/predict", formData, saveEvent);
    return normalizeFallResult(res.data, selectedFile.name);
  };

  const handlePredictWholeFile = async () => {
    if (!selectedFile) {
      setMessage("먼저 CSV 파일을 선택하세요.");
      return;
    }

    realtimeRunningRef.current = false;

    setLoading(true);
    setRealtimeRunning(false);
    setMessage("");

    try {
      const normalizedResult = await runWholeFileAggregatePredict(true);

      setPredictResult(normalizedResult);
      setSelectedEventId(null);

      if (normalizedResult.window_results?.length > 0) {
        setLiveResults([...normalizedResult.window_results].reverse());
        setLivePage(1);
      }

      if (normalizedResult.status === "Error") {
        setMessage(`예측 오류: ${normalizedResult.message}`);
      } else if (normalizedResult.status === "Fall Alert") {
        setMessage(
          "전체 CSV를 10프레임 단위로 집계한 결과 낙상 알림이 감지되었습니다.",
        );
      } else {
        setMessage(
          "전체 CSV를 10프레임 단위로 집계한 결과 정상 행동으로 판단되었습니다.",
        );
      }

      await refreshAll();
    } catch (err) {
      setMessage(`예측 요청 실패: ${getErrorMessage(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRealtimePredict = async () => {
    if (!selectedFile || !csvText) {
      setMessage("먼저 CSV 파일을 선택하세요.");
      return;
    }

    if (frameChunks.length === 0) {
      setMessage("10프레임 단위로 나눌 수 있는 frame 데이터가 없습니다.");
      return;
    }

    realtimeRunningRef.current = true;

    setLoading(true);
    setRealtimeRunning(true);
    setLiveResults([]);
    setLivePage(1);
    setPredictResult(null);
    setSelectedEventId(null);
    setCurrentChunkIndex(0);
    setMessage("10프레임 단위 실시간 시뮬레이션을 시작합니다.");

    let firstFall = null;

    for (let i = 0; i < frameChunks.length; i += 1) {
      if (!realtimeRunningRef.current) break;

      const chunk = frameChunks[i];
      setCurrentChunkIndex(i);

      try {
        const result = await predictCsvChunk(chunk);

        setPredictResult(result);
        setLiveResults((prev) => [result, ...prev]);
        setLivePage(1);

        if (result.status === "Fall Alert" && !firstFall) {
          firstFall = result;
          setMessage(
            `낙상 의심 구간 감지: frame ${result.frame_start} ~ ${result.frame_end}`,
          );
        }
      } catch (err) {
        const fallbackResult = normalizeFallResult(
          {
            status: "Error",
            alert: false,
            message: `이 10프레임 구간 분석 실패: ${getErrorMessage(err)}`,
            file_name: chunk.fileName,
            frame_start: chunk.startFrame,
            frame_end: chunk.endFrame,
            frame_count: chunk.frameCount,
            point_count: chunk.rowCount,
          },
          chunk.fileName,
        );

        const safeResult = {
          ...fallbackResult,
          chunk_index: chunk.index,
          frame_start: chunk.startFrame,
          frame_end: chunk.endFrame,
          frame_count: chunk.frameCount,
          row_count: chunk.rowCount,
        };

        setPredictResult(safeResult);
        setLiveResults((prev) => [safeResult, ...prev]);
        setLivePage(1);
      }

      await sleep(REALTIME_INTERVAL_MS);
    }

    realtimeRunningRef.current = false;
    setRealtimeRunning(false);

    try {
      const finalResult = await runWholeFileAggregatePredict(true);
      setPredictResult(finalResult);

      if (finalResult.window_results?.length > 0) {
        setLiveResults([...finalResult.window_results].reverse());
        setLivePage(1);
      }

      if (finalResult.status === "Fall Alert") {
        setMessage(
          `실시간 시뮬레이션 완료. 최종 낙상 위험도는 ${getDisplayFallPercent(
            finalResult,
          )}%이며, 가장 위험한 구간은 frame ${finalResult.frame_start} ~ ${
            finalResult.frame_end
          }입니다.`,
        );
      } else {
        setMessage(
          "실시간 시뮬레이션 완료. Fall Alert 구간이 감지되지 않았습니다.",
        );
      }

      await refreshAll();
    } catch (err) {
      if (firstFall) {
        setMessage(
          `실시간 시뮬레이션 완료. 첫 낙상 의심 구간은 frame ${firstFall.frame_start} ~ ${firstFall.frame_end}입니다.`,
        );
      } else {
        setMessage(`최종 집계 요청 실패: ${getErrorMessage(err)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStopRealtime = () => {
    realtimeRunningRef.current = false;
    setRealtimeRunning(false);
    setLoading(false);
    setMessage("실시간 시뮬레이션을 중지했습니다.");
  };

  const handleDeleteEvents = async () => {
    const ok = window.confirm("저장된 낙상 알림 로그를 모두 삭제할까요?");
    if (!ok) return;

    try {
      await apiDelete("/events");
      await refreshAll();
      setPredictResult(null);
      setSelectedEventId(null);
      setMessage("이벤트 로그를 삭제했습니다.");
    } catch (err) {
      setMessage(`이벤트 삭제 실패: ${getErrorMessage(err)}`);
    }
  };

  const handleSelectEvent = (event) => {
    const eventId = getEventId(event);

    const historyResult = {
      ...normalizeFallResult(event, event.file_name || event.filename || "-"),
      db_saved: true,
      db_message: "저장된 과거 낙상 알림 이벤트입니다.",
    };

    realtimeRunningRef.current = false;

    setSelectedFile(null);
    setCsvText("");
    setFrameChunks([]);
    setLiveResults(
      historyResult.window_results
        ? [...historyResult.window_results].reverse()
        : [],
    );
    setLivePage(1);
    setCurrentChunkIndex(-1);
    setRealtimeRunning(false);
    setPredictResult(historyResult);
    setSelectedEventId(eventId);
    setMessage("과거 낙상 알림 로그를 불러왔습니다.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const apiConnected = Boolean(serverInfo);
  const modelReady = Boolean(fallInfo?.model_exists);
  const mongoConnected = Boolean(fallInfo?.mongo_connected);

  return (
    <div className="page">
      <header className="dashboard-header">
        <div>
          <h1>독거노인 낙상 감지 관제 대시보드</h1>
          <p>
            전체 파일과 10프레임 실시간 분석 모두 같은 기준으로 낙상 위험도를
            계산합니다.
          </p>
        </div>

        <button className="refresh-btn" onClick={refreshAll}>
          연결 다시 확인
        </button>
      </header>

      <section className="fall-summary-grid">
        <div className="fall-summary-card api-card">
          <p>API 상태</p>
          <strong>{apiConnected ? "정상 연결" : "연결 안 됨"}</strong>
          <span>
            {apiConnected
              ? "FastAPI 서버와 연결되었습니다."
              : "백엔드 서버를 먼저 실행하세요."}
          </span>
        </div>

        <div className="fall-summary-card">
          <p>낙상 모델</p>
          <strong>{modelReady ? "로드 가능" : "확인 필요"}</strong>
          <span>
            {modelReady
              ? "models 폴더에서 pkl 파일을 찾았습니다."
              : "models 폴더의 pkl 파일을 확인하세요."}
          </span>
        </div>

        <div className="fall-summary-card mongo-card">
          <p>MongoDB</p>
          <strong>{mongoConnected ? "연결됨" : "저장 비활성"}</strong>
          <span>MongoDB가 꺼져 있어도 예측은 가능합니다.</span>
        </div>

        <div className="fall-summary-card">
          <p>분석 단위</p>
          <strong>{FRAME_WINDOW_SIZE}프레임</strong>
          <span>
            10FPS 기준 약 {((FRAME_WINDOW_SIZE / FPS) * 2).toFixed(1)}초
          </span>
        </div>

        <div className="fall-summary-card danger-soft">
          <p>보호자 알림</p>
          <strong>{stats?.fall_alert_count ?? 0}</strong>
          <span>즉시 확인 필요 건수</span>
        </div>
      </section>

      <main className="main-grid">
        <section className="card upload-card">
          <h2>CSV 낙상 예측 테스트</h2>
          <p className="desc">
            CSV 파일을 업로드하면 frame 번호 기준으로 10프레임씩 나누어
            실시간처럼 분석합니다.
          </p>

          <div className="upload-box">
            <input type="file" accept=".csv" onChange={handleFileChange} />

            <button onClick={handleRealtimePredict} disabled={loading}>
              {realtimeRunning ? "실행 중..." : "10프레임 실시간 실행"}
            </button>

            <button onClick={handlePredictWholeFile} disabled={loading}>
              전체 파일 예측
            </button>

            <button
              type="button"
              onClick={handleStopRealtime}
              disabled={!realtimeRunning}
            >
              중지
            </button>
          </div>

          {selectedFile && (
            <p className="selected-file">선택 파일: {selectedFile.name}</p>
          )}

          {message && <div className="message">{message}</div>}

          <RealtimePanel
            selectedFile={selectedFile}
            frameChunks={frameChunks}
            realtimeRunning={realtimeRunning}
            currentChunkIndex={currentChunkIndex}
            liveResults={liveResults}
            livePage={livePage}
            setLivePage={setLivePage}
          />

          <ResultCard result={predictResult} />

          {(predictResult || liveResults.length > 0) && (
            <div className="analysis-layout">
              <div className="analysis-left">
                {predictResult && <ProbabilityChart result={predictResult} />}

                <FallScoreChart
                  result={predictResult}
                  liveResults={liveResults}
                />
              </div>

              <div className="analysis-right">
                {predictResult && <EvidenceChart result={predictResult} />}
              </div>
            </div>
          )}
        </section>

        <aside className="side-panel">
          <RecentEventChart events={pagedEvents} />

          <section className="card events-card">
            <div className="section-title-row">
              <div>
                <h2>최근 낙상 알림 로그</h2>
                <p className="desc">
                  최종 집계된 Fall Alert 이벤트만 표시합니다.
                </p>
              </div>

              <button className="delete-btn" onClick={handleDeleteEvents}>
                로그 삭제
              </button>
            </div>

            <div className="table-wrap history-table-wrap">
              <table className="history-log-table">
                <thead>
                  <tr>
                    <th>시간</th>
                    <th>상태</th>
                    <th>알림</th>
                    <th>파일명</th>
                    <th>위험도</th>
                  </tr>
                </thead>

                <tbody>
                  {pagedEvents.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="empty">
                        저장된 낙상 알림이 없습니다.
                      </td>
                    </tr>
                  ) : (
                    pagedEvents.map((event) => {
                      const normalizedEvent = normalizeFallResult(event);
                      const eventId = getEventId(normalizedEvent);
                      const isSelected = selectedEventId === eventId;

                      return (
                        <tr
                          key={eventId}
                          className={`clickable-row ${
                            isSelected ? "selected-row" : ""
                          }`}
                          onClick={() => handleSelectEvent(normalizedEvent)}
                        >
                          <td className="time-cell">
                            {formatDateTime(normalizedEvent.created_at)}
                          </td>

                          <td className="status-cell">
                            <StatusBadge status={normalizedEvent.status} />
                          </td>

                          <td className="alert-cell">
                            <AlertText alert={normalizedEvent.alert} />
                          </td>

                          <td className="file-cell">
                            {normalizedEvent.file_name}
                          </td>

                          <td className="prob-cell">
                            {getDisplayFallPercent(normalizedEvent)}%
                          </td>
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
        </aside>
      </main>

      {predictResult && <BehaviorAnalysisCard result={predictResult} />}
    </div>
  );
}

export default FallDashboard;

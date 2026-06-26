import { useNavigate } from "react-router-dom";
import "./Home.css";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      <section className="home-hero">
        <div className="hero-left">
          <div className="service-badge">
            <span></span>
            Smart Care AI Integrated Monitoring Platform
          </div>

          <h1>
            독거노인 안전 관리를 위한
            <br />
            AI 통합 관제 플랫폼
          </h1>

          <p className="hero-desc">
            mmWave 센서 기반 낙상 감지, 생활 행동 분석, 생체신호 이상탐지를
            하나의 통합 관제 화면에서 확인할 수 있습니다. 관제 담당자는 위험
            상황, 주의 상태, 보호자 알림 필요 여부를 한눈에 파악하고 빠르게
            대응할 수 있습니다.
          </p>

          <div className="hero-buttons single">
            <button onClick={() => navigate("/integrated")}>
              통합관제 열기
            </button>
          </div>

          <div className="hero-tags">
            <span>통합 관제</span>
            <span>mmWave 낙상 감지</span>
            <span>이상행동 분석</span>
            <span>생체신호 이상탐지</span>
            <span>보호자 알림</span>
          </div>
        </div>

        <div className="hero-right">
          <div className="monitor-card">
            <div className="monitor-header">
              <div>
                <p>CARE CONTROL</p>
                <h2>통합 관제 흐름</h2>
              </div>
              <span className="ready-badge">Integrated AI</span>
            </div>

            <div className="monitor-flow">
              <FlowItem
                number="01"
                title="센서 데이터 수집"
                desc="낙상, 이상행동, 생체신호 데이터를 각각 수집합니다."
              />

              <FlowItem
                number="02"
                title="AI 모델별 분석"
                desc="낙상 모델, 행동 분석 모델, 생체신호 이상탐지 모델을 독립적으로 실행합니다."
              />

              <FlowItem
                number="03"
                title="통합 위험 판단"
                desc="각 분석 결과를 통합하여 정상, 주의, 위험, Fall Alert 상태를 판단합니다."
              />

              <FlowItem
                number="04"
                title="관제 및 보호자 확인"
                desc="위험 상황은 통합관제 화면에 표시하고 보호자 알림 필요 상태로 관리합니다."
              />
            </div>
          </div>
        </div>
      </section>

      <section className="feature-section">
        <div className="section-title">
          <p>MAIN FEATURES</p>
          <h2>주요 기능</h2>
          <span>
            낙상 감지, 이상행동 분석, 생체신호 이상탐지를 개별로 확인할 수 있고,
            핵심 결과는 통합관제에서 함께 관리할 수 있도록 구성했습니다.
          </span>
        </div>

        <div className="feature-grid">
          <FeatureCard
            icon="01"
            title="통합관제 대시보드"
            desc="낙상, 이상행동, 생체신호 분석 결과를 하나의 화면에서 종합적으로 확인합니다."
          />

          <FeatureCard
            icon="02"
            title="낙상 감지"
            desc="mmWave CSV 데이터를 분석하여 낙상 위험도와 Fall Alert 여부를 판단합니다."
          />

          <FeatureCard
            icon="03"
            title="이상행동 분석"
            desc="무활동, 배회, 위험 행동, 생활 상태를 분석하여 주의 또는 위험 상태를 표시합니다."
          />

          <FeatureCard
            icon="04"
            title="생체신호 분석"
            desc="생체신호 데이터를 기반으로 복원 오차와 임계값을 비교하여 이상 여부를 판단합니다."
          />
        </div>
      </section>

      <section className="scenario-section">
        <div className="scenario-card">
          <div className="scenario-title">
            <p>INTEGRATED SCENARIO</p>
            <h2>통합 분석 대상</h2>
          </div>

          <div className="scenario-grid">
            <ScenarioItem
              title="낙상 사고"
              desc="높이 변화, 속도 변화, 낙상 이후 움직임을 분석합니다."
            />

            <ScenarioItem
              title="이상행동"
              desc="배회, 무활동, 빠른 자세 변화 등 생활 행동의 위험 구간을 판단합니다."
            />

            <ScenarioItem
              title="생체신호 이상"
              desc="복원 오차, 임계값, 통계 특징을 기준으로 주의 상태를 감지합니다."
            />

            <ScenarioItem
              title="보호자 알림 필요"
              desc="Fall Alert 또는 위험 상태가 발생하면 보호자 확인 필요 상태로 표시합니다."
            />
          </div>
        </div>
      </section>
    </div>
  );
}

function FlowItem({ number, title, desc }) {
  return (
    <div className="flow-item">
      <strong>{number}</strong>
      <div>
        <h3>{title}</h3>
        <p>{desc}</p>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="feature-card">
      <span>{icon}</span>
      <h3>{title}</h3>
      <p>{desc}</p>
    </div>
  );
}

function ScenarioItem({ title, desc }) {
  return (
    <div className="scenario-item">
      <h3>{title}</h3>
      <p>{desc}</p>
    </div>
  );
}

export default Home;

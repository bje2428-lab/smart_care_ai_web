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
            Smart Care AI Monitoring Platform
          </div>

          <h1>
            독거노인 안전 관리를 위한
            <br />
            AI 협업 관제 플랫폼
          </h1>

          <p className="hero-desc">
            mmWave 센서 데이터를 기반으로 낙상 사고뿐만 아니라 장시간 움직임
            없음, 반복 배회와 같은 이상행동을 분석하여 보호자와 관제 담당자가
            빠르게 대응할 수 있도록 지원하는 스마트 돌봄 관제 서비스입니다.
          </p>

          <div className="hero-buttons">
            <button onClick={() => navigate("/fall")}>
              관제 대시보드 열기
            </button>

            <button className="outline" onClick={() => navigate("/fall")}>
              CSV 예측 테스트
            </button>

            <button className="outline" onClick={() => navigate("/abnormal")}>
              이상행동 분석 보기
            </button>
          </div>

          <div className="hero-tags">
            <span>낙상 감지</span>
            <span>이상행동 분석</span>
            <span>비영상 센서 기반</span>
            <span>실시간 알림</span>
          </div>
        </div>

        <div className="hero-right">
          <div className="monitor-card">
            <div className="monitor-header">
              <div>
                <p>CARE CONTROL</p>
                <h2>안전 관제 흐름</h2>
              </div>
              <span className="ready-badge">AI 분석</span>
            </div>

            <div className="monitor-flow">
              <FlowItem
                number="01"
                title="센서 데이터 수집"
                desc="mmWave 센서로 움직임 데이터를 수집합니다."
              />
              <FlowItem
                number="02"
                title="행동 패턴 분석"
                desc="낙상, 무활동, 반복 배회 등 위험 행동을 분석합니다."
              />
              <FlowItem
                number="03"
                title="위험 상황 판단"
                desc="AI 모델이 이상 여부와 알림 필요성을 판단합니다."
              />
              <FlowItem
                number="04"
                title="관제 및 보호자 연계"
                desc="관제 화면에서 결과를 확인하고 대응할 수 있습니다."
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
            실제 돌봄 상황에서 필요한 낙상 감지, 이상행동 분석, 알림, 관제
            기능을 하나의 서비스 흐름으로 구성했습니다.
          </span>
        </div>

        <div className="feature-grid">
          <FeatureCard
            icon="01"
            title="낙상 감지"
            desc="mmWave 센서 CSV 데이터를 분석하여 넘어짐으로 의심되는 상황을 감지합니다."
          />
          <FeatureCard
            icon="02"
            title="이상행동 분석"
            desc="장시간 움직임 없음, 반복 배회, 비정상적인 움직임 패턴을 분석할 수 있도록 확장합니다."
          />
          <FeatureCard
            icon="03"
            title="비영상 기반 안전 관리"
            desc="카메라 영상 대신 센서 데이터를 활용하여 개인정보 부담을 줄입니다."
          />
          <FeatureCard
            icon="04"
            title="협업 관제"
            desc="예측 결과와 이벤트 로그를 관제 화면에서 확인하고 보호자 대응 흐름과 연결합니다."
          />
        </div>
      </section>

      <section className="scenario-section">
        <div className="scenario-card">
          <div>
            <p>DETECTION SCENARIO</p>
            <h2>분석 대상 행동</h2>
          </div>

          <div className="scenario-grid">
            <ScenarioItem
              title="낙상 사고"
              desc="갑작스러운 자세 변화와 속도 변화를 분석"
            />
            <ScenarioItem
              title="장시간 무활동"
              desc="일정 시간 이상 움직임이 없는 상태 확인"
            />
            <ScenarioItem
              title="반복 배회"
              desc="같은 구간을 반복적으로 이동하는 패턴 분석"
            />
            <ScenarioItem
              title="위험 알림"
              desc="이상 상황 발생 시 관제 화면에서 빠르게 확인"
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

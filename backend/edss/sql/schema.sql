-- Engineering Decision Intelligence System — Full Database Schema
-- PostgreSQL 16+

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ═══════════════════════════════════════════════════════════════════
-- USERS
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE,
    name TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- DECISION CASES (core table)
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS decision_cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    domain TEXT NOT NULL,
    description TEXT DEFAULT '',
    problem_type TEXT,
    decision_maker TEXT,
    objective_direction TEXT CHECK (objective_direction IN ('maximize', 'minimize')),
    unit TEXT,
    time_horizon TEXT,
    risk_preference TEXT CHECK (risk_preference IN ('neutral', 'averse', 'seeking')) DEFAULT 'neutral',
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- PROBLEM VERSIONING (audit trail)
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS problem_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    snapshot JSONB NOT NULL,
    change_reason TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS problem_inputs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    input_type TEXT NOT NULL, -- 'csv', 'json', 'text', 'pdf'
    content JSONB NOT NULL DEFAULT '{}',
    filename TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- CLASSIFICATION
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS problem_classifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    problem_type TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    scores JSONB NOT NULL DEFAULT '{}',
    reason TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- MATHEMATICAL MODEL COMPONENTS
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS decision_variables (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    variable_type TEXT CHECK (variable_type IN ('continuous', 'integer', 'binary')),
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    unit TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS objective_functions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    sense TEXT CHECK (sense IN ('maximize', 'minimize')),
    expression TEXT,
    coefficients JSONB NOT NULL DEFAULT '{}',
    constant DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS constraints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    expression TEXT,
    coefficients JSONB NOT NULL DEFAULT '{}',
    operator TEXT CHECK (operator IN ('<=', '>=', '=')),
    rhs DOUBLE PRECISION NOT NULL,
    resource TEXT,
    unit TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- DECISION UNDER UNCERTAINTY
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS alternatives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    attributes JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS states_of_nature (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    probability DOUBLE PRECISION CHECK (probability >= 0),
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS probability_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    model_type TEXT NOT NULL, -- 'prior', 'posterior', 'test_sensitivity'
    parameters JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS payoff_matrix (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    alternative_id UUID REFERENCES alternatives(id) ON DELETE CASCADE,
    state_id UUID REFERENCES states_of_nature(id) ON DELETE CASCADE,
    payoff DOUBLE PRECISION NOT NULL,
    cost DOUBLE PRECISION DEFAULT 0,
    loss DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- NETWORK / GRAPH PROBLEMS
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS network_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    node_type TEXT DEFAULT 'intermediate', -- 'source', 'sink', 'transshipment', 'intermediate'
    supply DOUBLE PRECISION DEFAULT 0,
    demand DOUBLE PRECISION DEFAULT 0,
    label TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS network_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    cost DOUBLE PRECISION DEFAULT 0,
    capacity DOUBLE PRECISION,
    flow DOUBLE PRECISION DEFAULT 0,
    is_directed BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- SIMULATION
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS simulation_variables (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    distribution TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    unit TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- SOLVER RESULTS & ANALYSIS
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS solver_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    solver_type TEXT NOT NULL,
    status TEXT NOT NULL, -- 'optimal', 'feasible', 'infeasible', 'unbounded', 'computed'
    objective_value DOUBLE PRECISION,
    solution JSONB NOT NULL DEFAULT '{}',
    solver_log JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS solver_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    run_id UUID REFERENCES solver_runs(id) ON DELETE CASCADE,
    solver_type TEXT NOT NULL,
    objective_value DOUBLE PRECISION,
    solution JSONB NOT NULL DEFAULT '{}',
    shadow_prices JSONB NOT NULL DEFAULT '{}',
    slacks JSONB NOT NULL DEFAULT '{}',
    binding_constraints JSONB DEFAULT '[]',
    recommendation TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sensitivity_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES solver_runs(id) ON DELETE CASCADE,
    parameter_name TEXT NOT NULL,
    parameter_type TEXT, -- 'rhs', 'objective_coefficient', 'probability'
    base_value DOUBLE PRECISION,
    range_low DOUBLE PRECISION,
    range_high DOUBLE PRECISION,
    shadow_price DOUBLE PRECISION,
    impact DOUBLE PRECISION,
    tornado_data JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS risk_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES solver_runs(id) ON DELETE CASCADE,
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    var_value DOUBLE PRECISION,
    cvar_value DOUBLE PRECISION,
    probability_of_loss DOUBLE PRECISION,
    worst_case DOUBLE PRECISION,
    distribution_data JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS information_value_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES solver_runs(id) ON DELETE CASCADE,
    evwpi DOUBLE PRECISION,
    evwopi DOUBLE PRECISION,
    evpi DOUBLE PRECISION,
    evi DOUBLE PRECISION,
    max_price_for_info DOUBLE PRECISION,
    should_buy_info BOOLEAN,
    test_analysis JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- REPORTS
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES decision_cases(id) ON DELETE CASCADE,
    format TEXT DEFAULT 'markdown', -- 'markdown', 'pdf', 'html'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- KNOWLEDGE LAYER
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS knowledge_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    source_type TEXT DEFAULT 'pdf',
    total_pages INTEGER,
    ingested_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    page_number INTEGER,
    section_title TEXT,
    content TEXT NOT NULL,
    concepts TEXT[],
    embedding BYTEA,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_concepts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    definition TEXT,
    formula TEXT,
    source_ref TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_concept UUID REFERENCES knowledge_concepts(id) ON DELETE CASCADE,
    target_concept UUID REFERENCES knowledge_concepts(id) ON DELETE CASCADE,
    relation TEXT NOT NULL, -- 'HAS_METHOD', 'IS_A', 'USES', 'SOLVED_BY', etc.
    created_at TIMESTAMP DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_cases_user ON decision_cases(user_id);
CREATE INDEX IF NOT EXISTS idx_cases_type ON decision_cases(problem_type);
CREATE INDEX IF NOT EXISTS idx_solver_results_case ON solver_results(case_id);
CREATE INDEX IF NOT EXISTS idx_versions_case ON problem_versions(case_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source ON knowledge_chunks(source_id);

CREATE TABLE assessments (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    event_id TEXT,
    building_uid TEXT,
    coverage_id TEXT,
    imagery_acquired_at TIMESTAMPTZ,
    location GEOMETRY(POINT, 4326),
    filename TEXT NOT NULL,
    pre_filename TEXT,
    post_image_path TEXT NOT NULL,
    pre_image_path TEXT,
    mask_path TEXT NOT NULL,
    post_image_metadata JSONB NOT NULL,
    pre_image_metadata JSONB,
    model_version TEXT NOT NULL,
    segmentation_model_version TEXT,
    prediction TEXT NOT NULL CHECK (prediction IN ('no-damage', 'minor-damage', 'major-damage', 'destroyed')),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    probabilities JSONB NOT NULL,
    severity_score DOUBLE PRECISION NOT NULL CHECK (severity_score BETWEEN 0 AND 100),
    damage_area_ratio DOUBLE PRECISION NOT NULL CHECK (damage_area_ratio BETWEEN 0 AND 1),
    segmentation_method TEXT NOT NULL,
    model_disagreement JSONB NOT NULL,
    decision_rules JSONB NOT NULL,
    image_quality_warnings JSONB NOT NULL,
    review_reasons JSONB NOT NULL,
    priority_score DOUBLE PRECISION NOT NULL CHECK (priority_score BETWEEN 0 AND 100),
    priority_level TEXT NOT NULL,
    priority_components JSONB NOT NULL,
    priority_formula TEXT NOT NULL,
    priority_rationale TEXT NOT NULL,
    review_status TEXT NOT NULL CHECK (review_status IN ('unreviewed', 'pending', 'confirmed', 'corrected', 'unresolved')),
    human_damage_class TEXT CHECK (human_damage_class IS NULL OR human_damage_class IN ('no-damage', 'minor-damage', 'major-damage', 'destroyed')),
    reviewer TEXT,
    review_notes TEXT,
    reviewed_at TIMESTAMPTZ,
    CHECK ((human_damage_class IS NULL) OR review_status IN ('confirmed', 'corrected')),
    CHECK ((location IS NULL) OR (ST_Y(location) BETWEEN -90 AND 90 AND ST_X(location) BETWEEN -180 AND 180))
);

CREATE INDEX assessments_created_at_idx ON assessments (created_at DESC);
CREATE INDEX assessments_event_idx ON assessments (event_id);
CREATE INDEX assessments_prediction_idx ON assessments (prediction);
CREATE INDEX assessments_review_status_idx ON assessments (review_status);
CREATE INDEX assessments_location_gist_idx ON assessments USING GIST (location);

CREATE TABLE assessment_reviews (
    id BIGSERIAL PRIMARY KEY,
    assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('confirmed', 'corrected', 'unresolved')),
    human_damage_class TEXT CHECK (human_damage_class IS NULL OR human_damage_class IN ('no-damage', 'minor-damage', 'major-damage', 'destroyed')),
    reviewer TEXT NOT NULL,
    notes TEXT,
    reviewed_at TIMESTAMPTZ NOT NULL,
    CHECK ((status IN ('confirmed', 'corrected') AND human_damage_class IS NOT NULL) OR (status = 'unresolved' AND human_damage_class IS NULL))
);

CREATE INDEX assessment_reviews_assessment_idx ON assessment_reviews (assessment_id, reviewed_at DESC);

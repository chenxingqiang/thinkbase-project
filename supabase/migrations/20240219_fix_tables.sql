-- Drop existing tables
DROP TABLE IF EXISTS overleaf_templates_detail;
DROP TABLE IF EXISTS overleaf_templates_general;

-- Create general templates table
CREATE TABLE overleaf_templates_general (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    link TEXT UNIQUE NOT NULL,
    author TEXT,
    preview_image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create detail templates table
CREATE TABLE overleaf_templates_detail (
    id TEXT PRIMARY KEY,
    link TEXT UNIQUE NOT NULL REFERENCES overleaf_templates_general(link) ON DELETE CASCADE,
    last_updated TEXT,
    license TEXT,
    abstract TEXT,
    open_template_link TEXT,
    view_pdf_link TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_general_link ON overleaf_templates_general(link);
CREATE INDEX idx_detail_link ON overleaf_templates_detail(link);

-- Enable RLS
ALTER TABLE overleaf_templates_general ENABLE ROW LEVEL SECURITY;
ALTER TABLE overleaf_templates_detail ENABLE ROW LEVEL SECURITY;

-- Drop existing policies
DROP POLICY IF EXISTS "Allow read access for authenticated users" ON overleaf_templates_general;
DROP POLICY IF EXISTS "Allow write access for authenticated users" ON overleaf_templates_general;
DROP POLICY IF EXISTS "Allow read access for authenticated users" ON overleaf_templates_detail;
DROP POLICY IF EXISTS "Allow write access for authenticated users" ON overleaf_templates_detail;

-- Create policies for general table
CREATE POLICY "Allow read access for authenticated users" ON overleaf_templates_general
    FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Allow insert access for authenticated users" ON overleaf_templates_general
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow update access for authenticated users" ON overleaf_templates_general
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Create policies for detail table
CREATE POLICY "Allow read access for authenticated users" ON overleaf_templates_detail
    FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Allow insert access for authenticated users" ON overleaf_templates_detail
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow update access for authenticated users" ON overleaf_templates_detail
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true); 
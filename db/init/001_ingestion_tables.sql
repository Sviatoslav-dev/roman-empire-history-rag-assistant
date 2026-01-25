-- Ingestion metadata tables (separate from Qdrant data)

create table if not exists ingestion_article (
    id bigserial primary key,
    title text not null unique,
    url text,
    local_path text,
    symbols_number integer,
    citations_number integer,
    has_problem_or_update_box boolean,
    is_english_available boolean,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists ingestion_image (
    id bigserial primary key,
    url text not null unique,
    licence text,
    local_path text,
    filename text,
    extension text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_ingestion_article_url on ingestion_article(url);
create index if not exists idx_ingestion_image_filename on ingestion_image(filename);


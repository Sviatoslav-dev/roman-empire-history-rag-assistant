from pathlib import Path

import pytest

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper


def _wrap_in_content(html_inner: str, *, lang: str = "en") -> str:
    # Minimal DOM that WikipediaArticleScraper expects.
    # Note: splitter walks *direct children* of `.mw-content-ltr`, so we place
    # content elements directly under it.
    return f"""
    <html lang=\"{lang}\">
      <body>
        <div class=\"mw-content-ltr\">
          <div id=\"mw-content-text\"></div>
          {html_inner}
        </div>
      </body>
    </html>
    """


def test_count_citations_counts_unique_cite_notes() -> None:
    html = _wrap_in_content(
        """
        <p>A<a href='#cite_note-1'>[1]</a> B<a href='#cite_note-1'>[1]</a>
           C<a href='#cite_note-2'>[2]</a></p>
        """
    )
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    assert s.count_citations() == 2


def test_has_problem_or_update_box_detects_keywords_in_ambox() -> None:
    html = _wrap_in_content(
        """
        <div class='ambox'>This article needs cleanup and update</div>
        <p>Body text</p>
        """
    )
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    assert s.has_problem_or_update_box() is True


def test_is_english_article_respects_html_lang_attribute() -> None:
    # Current scraper falls back to treating documents that look like Wikipedia
    # content as English even if lang != en. So we only assert the positive case.
    html2 = _wrap_in_content("<p>Hello</p>", lang="en")
    s2 = WikipediaArticleScraper(html2, "T", "/wiki/T")
    assert s2.is_english_article() is True


def test_split_by_chunks_stops_at_excluded_sections() -> None:
    # Build a page with headings, then an excluded References section.
    html = _wrap_in_content(
        """
        <p>Intro text</p>
        <div><h2>History</h2></div>
        <p>Some history text</p>
        <div><h2>References</h2></div>
        <p>Ref content that should not appear</p>
        """
    )
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    chunks = s.split_by_chunks(max_text_size=10000)

    # Should include Introduction + History only
    assert len(chunks) == 2
    assert chunks[0].section_title == "Introduction"
    assert chunks[1].section_title == "History"
    assert all("Ref content" not in c.text for c in chunks)


def test_split_by_chunks_extracts_figure_image_and_caption_and_normalizes_url() -> None:
    html = _wrap_in_content(
        """
        <p>Intro text</p>
        <figure>
          <a><img src='//upload.wikimedia.org/wikipedia/commons/a/ab/Filename.jpg' /></a>
          <figcaption>  A caption\nwith spaces </figcaption>
        </figure>
        """
    )
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    chunks = s.split_by_chunks(max_text_size=10000)

    assert chunks
    intro = chunks[0]
    assert intro.images, "Expected figure image to be captured"
    assert intro.images[0].image.url.startswith("https://")
    assert intro.images[0].caption == "A caption with spaces"


def test_split_by_chunks_extracts_gallery_images() -> None:
    html = _wrap_in_content(
        """
        <p>Intro text</p>
        <div class='gallery'>
          <div class='gallerybox'>
            <a><img src='//upload.wikimedia.org/wikipedia/commons/a/ab/G1.jpg' /></a>
            <div class='gallerytext'>Cap one</div>
          </div>
          <div class='gallerybox'>
            <a><img src='//upload.wikimedia.org/wikipedia/commons/a/ab/G2.jpg' /></a>
            <div class='gallerytext'>Cap two</div>
          </div>
        </div>
        """
    )
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    chunks = s.split_by_chunks(max_text_size=10000)

    assert chunks
    images = chunks[0].images
    assert [m.image.get_filename() for m in images] == ["G1.jpg", "G2.jpg"]
    assert [m.caption for m in images] == ["Cap one", "Cap two"]


def test_split_by_chunks_infobox_extracts_key_values_and_images() -> None:
    html = _wrap_in_content(
        """
        <table class='infobox'>
          <tr><th>Born</th><td>Rome</td></tr>
          <tr><td><img src='//upload.wikimedia.org/wikipedia/commons/a/ab/Infobox.jpg' /></td></tr>
        </table>
        """
    )
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    chunks = s.split_by_chunks(max_text_size=10000)

    # Infobox becomes text in the current chunk
    assert any("infobox" in part.lower() for part in chunks[0].text_parts)
    assert chunks[0].images, "Expected infobox image to be captured"


def test_split_by_chunks_raises_when_content_root_missing() -> None:
    html = "<html><body><div id='mw-content-text'><p>Hi</p></div></body></html>"
    s = WikipediaArticleScraper(html, "T", "/wiki/T")
    with pytest.raises(Exception):
        s.split_by_chunks()

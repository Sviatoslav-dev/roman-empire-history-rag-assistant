from data_ingestion.scraper.wikipedia_category_scraper import WikipediaCategoryScraper


def _wrap_in_category_pages(inner: str) -> str:
    # Minimal DOM that WikipediaCategoryScraper expects.
    return f"""
    <html>
      <body>
        <div id='mw-pages'>
          {inner}
        </div>
      </body>
    </html>
    """


def test_extract_articles_from_category_extracts_only_article_links() -> None:
    html = _wrap_in_category_pages(
        """
        <div class='mw-category-group'>
          <a href='/wiki/Article_One'>Article one</a>
          <a href='/wiki/Article_Two'>Article two</a>
        </div>
        """
    )

    s = WikipediaCategoryScraper(html, "Category:Test")
    assert s.extract_articles_from_category() == {"/wiki/Article_One", "/wiki/Article_Two"}


def test_extract_articles_from_category_filters_namespace_links_with_colon() -> None:
    html = _wrap_in_category_pages(
        """
        <div class='mw-category-group'>
          <a href='/wiki/Article_One'>Article one</a>
          <a href='/wiki/Category:Something'>Category page</a>
          <a href='/wiki/File:Some_Image.jpg'>File page</a>
          <a href='/wiki/Help:Contents'>Help page</a>
        </div>
        """
    )

    s = WikipediaCategoryScraper(html, "Category:Test")
    assert s.extract_articles_from_category() == {"/wiki/Article_One"}


def test_extract_articles_from_category_decodes_url_escapes_before_filtering() -> None:
    # Encoded colon (%3A) should still be treated as a namespace link and filtered out.
    html = _wrap_in_category_pages(
        """
        <div class='mw-category-group'>
          <a href='/wiki/Category%3AEncoded'>Encoded category link</a>
          <a href='/wiki/Normal_Article'>Normal</a>
        </div>
        """
    )

    s = WikipediaCategoryScraper(html, "Category:Test")
    assert s.extract_articles_from_category() == {"/wiki/Normal_Article"}


def test_extract_articles_from_category_deduplicates_urls_across_groups() -> None:
    html = _wrap_in_category_pages(
        """
        <div class='mw-category-group'>
          <a href='/wiki/Article_One'>Article one</a>
        </div>
        <div class='mw-category-group'>
          <a href='/wiki/Article_One'>Article one duplicate</a>
          <a href='/wiki/Article_Two'>Article two</a>
        </div>
        """
    )

    s = WikipediaCategoryScraper(html, "Category:Test")
    assert s.extract_articles_from_category() == {"/wiki/Article_One", "/wiki/Article_Two"}


import pytest
from github_seo_analyzer.core.analyzer import SEOAnalyzer
from github_seo_analyzer.core.metrics import SEOMetrics

def test_analyzer_initialization():
    """Test SEO analyzer initialization with default settings."""
    analyzer = SEOAnalyzer()
    assert analyzer is not None
    assert isinstance(analyzer.metrics, SEOMetrics)

def test_analyze_repository_metadata():
    """Test analysis of repository metadata."""
    analyzer = SEOAnalyzer()
    repo_url = "https://github.com/chenxingqiang/APOS"
    results = analyzer.analyze_repository(repo_url)
    
    assert results is not None
    assert "title" in results
    assert "description" in results
    assert "keywords" in results
    assert "score" in results

def test_analyze_readme_content():
    """Test analysis of README.md content."""
    analyzer = SEOAnalyzer()
    repo_url = "https://github.com/chenxingqiang/APOS"
    results = analyzer.analyze_readme(repo_url)
    
    assert results is not None
    assert "content_score" in results
    assert "keyword_density" in results
    assert "suggestions" in results

def test_calculate_repository_score():
    """Test calculation of overall repository SEO score."""
    analyzer = SEOAnalyzer()
    repo_url = "https://github.com/chenxingqiang/APOS"
    score = analyzer.calculate_score(repo_url)
    
    assert isinstance(score, float)
    assert 0 <= score <= 100

def test_generate_improvement_suggestions():
    """Test generation of SEO improvement suggestions."""
    analyzer = SEOAnalyzer()
    repo_url = "https://github.com/chenxingqiang/APOS"
    suggestions = analyzer.generate_suggestions(repo_url)
    
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert all(isinstance(s, str) for s in suggestions)

def test_invalid_repository_url():
    """Test handling of invalid repository URL."""
    analyzer = SEOAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze_repository("invalid_url")

def test_repository_not_found():
    """Test handling of non-existent repository."""
    analyzer = SEOAnalyzer()
    with pytest.raises(Exception):
        analyzer.analyze_repository("https://github.com/chenxingqiang/nonexistent") 
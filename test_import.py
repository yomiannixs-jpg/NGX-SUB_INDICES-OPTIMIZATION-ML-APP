def test_import():
    import ngx_portfolio
    assert hasattr(ngx_portfolio, 'run_pipeline')

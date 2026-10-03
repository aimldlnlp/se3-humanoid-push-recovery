import matplotlib.pyplot as plt

from se3_whole_body_control.visualization.style import apply_style


def test_bundled_cff_font_exports_as_vector_glyphs_not_invalid_truetype(tmp_path):
    apply_style()
    figure, axis = plt.subplots()
    axis.set_xlabel('SE(3) geometric control')
    path = tmp_path / 'font.pdf'
    figure.savefig(path)
    plt.close(figure)
    content = path.read_bytes()
    assert content.startswith(b'%PDF-')
    assert b'/Subtype /Type3' in content and b'/CharProcs' in content
    assert b'/FontFile2' not in content

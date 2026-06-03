from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    app.run(timeout=180)

    print("exceptions:", len(app.exception))
    for exception in app.exception:
        print(exception.value)

    print("titles:", [item.value for item in app.title])
    print("subheaders:", [item.value for item in app.subheader])
    print("radios:", [(item.label, item.options, item.value) for item in app.radio])
    print("camera_inputs:", len(app.get("camera_input")))
    print("file_uploaders:", len(app.get("file_uploader")))


if __name__ == "__main__":
    main()


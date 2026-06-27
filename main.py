from src.core.kernel import QHNOKernel, QHNOConfig


def main():
    config = QHNOConfig(
        project_name="QHNO-Climate",
        version="0.1.0",
        debug=True
    )

    kernel = QHNOKernel(config)

    print(">>> QHNO BOOTING...")

    kernel.start()

    print(">>> QHNO SYSTEM ONLINE")


if __name__ == "__main__":
    main()

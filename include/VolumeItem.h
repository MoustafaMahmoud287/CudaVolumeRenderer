#pragma once
#include <QQuickFramebufferObject>
#include <QOpenGLFunctions>
#include <QtQml/qqml.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "VolumeTexture.cuh"


// gui thread

class VolumeItem : public QQuickFramebufferObject {
    Q_OBJECT // macro to support the use of signals and slots
    QML_ELEMENT // to be used on qml

    // properties exposed to qml for 3D rotation & min opecity
    Q_PROPERTY(float rotationX READ rotationX WRITE setRotationX NOTIFY rotationXChanged)
    Q_PROPERTY(float rotationY READ rotationY WRITE setRotationY NOTIFY rotationYChanged)
    Q_PROPERTY(float opecity READ opecity WRITE setOpecity NOTIFY opecityChanged)

public:
    explicit VolumeItem(QQuickItem *parent = nullptr);
    Renderer *createRenderer() const override; // will be casted later to our specefic renderer

    // getters
    float rotationX() const { return m_rotX; }
    float rotationY() const { return m_rotY; }
    float opecity() const { return m_opecity; }

    // setters
    void setRotationX(float angle);
    void setRotationY(float angle);
    void setOpecity(float opc);

signals:
    void rotationXChanged();
    void rotationYChanged();
    void opecityChanged();

private:
    float m_rotX = 0.0f;
    float m_rotY = 0.0f;
    float m_opecity = 0.0f;
};


// render thread

class VolumeRendererInternal : public QQuickFramebufferObject::Renderer, protected QOpenGLFunctions {
public:
    ~VolumeRendererInternal(); // Destructor to clean up GPU memory

    void render() override;
    QOpenGLFramebufferObject *createFramebufferObject(const QSize &size) override;

    // the handshake: copies data from gui thread to render thread safely
    void synchronize(QQuickFramebufferObject *item) override;

private:
    // engine components
    VolumeTexture volTex;
    bool isInitialized = false;

    // OpenGL resources
    GLuint pboID = 0;     // pixel buffer object (shared memory)
    GLuint textureID = 0; // the display texture

    // CUDA handle
    cudaGraphicsResource* cudaResource = nullptr;

    // state
    int width = 0;
    int height = 0;
    float3 m_boxSize = make_float3(1.0f, 1.0f, 1.0f);
    // render thread copies of the rotation angles
    float m_renderRotX = 0.0f;
    float m_renderRotY = 0.0f;
    float m_renderOpc = 0.0f;
};

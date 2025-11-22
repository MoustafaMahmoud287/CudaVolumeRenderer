#include "VolumeItem.h"
#include "VolumeLoader.h"
#include "VolumeRenderer.cuh" // For launchKernel static function
#include <QOpenGLFramebufferObject>
#include <QQuickWindow>
#include <QDebug>


// gui thread implementation


VolumeItem::VolumeItem(QQuickItem *parent) : QQuickFramebufferObject(parent) {
    // OpenGL renders upside-down relative to QML coordinates, so we flip it.
    setMirrorVertically(true);
}

void VolumeItem::setRotationX(float angle) {
    if (m_rotX != angle) {
        m_rotX = angle;
        emit rotationXChanged();
        update(); // redraw
    }
}

void VolumeItem::setRotationY(float angle) {
    if (m_rotY != angle) {
        m_rotY = angle;
        emit rotationYChanged();
        update(); // redraw
    }
}

void VolumeItem::setOpecity(float opc) {
    if (m_opecity != opc) {
        m_opecity = opc;
        emit opecityChanged();
        update(); // redraw
    }
}

QQuickFramebufferObject::Renderer* VolumeItem::createRenderer() const {
    return new VolumeRendererInternal();
}


// render thread implementation


// destructor: cleanup to prevent crashes on app exit
VolumeRendererInternal::~VolumeRendererInternal() {
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
    }
    if (pboID) {
        glDeleteBuffers(1, &pboID);
    }
    if (textureID) {
        glDeleteTextures(1, &textureID);
    }
}

QOpenGLFramebufferObject* VolumeRendererInternal::createFramebufferObject(const QSize &size) {
    QOpenGLFramebufferObjectFormat format;
    format.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
    return new QOpenGLFramebufferObject(size, format);
}

// synchronize: copy data safely while threads are locked
void VolumeRendererInternal::synchronize(QQuickFramebufferObject *item) {
    VolumeItem *guiItem = static_cast<VolumeItem*>(item);
    m_renderRotX = guiItem->rotationX();
    m_renderRotY = guiItem->rotationY();
    m_renderOpc = guiItem->opecity();
}

void VolumeRendererInternal::render() {
    // initialize OpenGL Functions
    initializeOpenGLFunctions();

    if (!isInitialized) {
        int w = 128, h = 256, d = 256;
        VolumeLoader loader(w, h, d);
        if (loader.load("maleHead.raw")) {
            volTex.loadVolume(loader.data, w, h, d);
            isInitialized = true;
            qDebug() << "[CUDA] Volume Data Loaded.";
        } else {
            qDebug() << "[ERROR] Could not load 'maleHead.raw'.";
            // visual error indicator: clear screen Red
            glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            return;
        }
        float maxd = std::max({w, h, d});
        m_boxSize.x = w / maxd;
        m_boxSize.y = h / maxd;
        m_boxSize.z = d / maxd;
    }

    // 3. handle resize (recreate buffers)
    QSize size = framebufferObject()->size();
    if (width != size.width() || height != size.height()) {
        width = size.width();
        height = size.height();

        // cleanup old resources if they exist
        if (pboID) {
            if (cudaResource) cudaGraphicsUnregisterResource(cudaResource);
            glDeleteBuffers(1, &pboID);
            glDeleteTextures(1, &textureID);
        }

        // A. Create OpenGL Texture
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // B. Create PBO (Shared Buffer)
        glGenBuffers(1, &pboID);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);

        // C. Register with CUDA
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess) qDebug() << "CUDA Register Error:" << cudaGetErrorString(err);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    // run CUDA kernel

    // A. Lock buffer for CUDA
    cudaGraphicsMapResources(1, &cudaResource, 0);

    // B. Get Device Pointer
    uchar4* d_ptr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &num_bytes, cudaResource);

    // C. Launch Kernel (Passing both rotation angles)
    VolumeRenderer::launchKernel(d_ptr, volTex.getTexture(), width, height, m_renderRotX, m_renderRotY, m_renderOpc, m_boxSize);

    // D. Unlock Buffer for OpenGL
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    // DRAW TO SCREEN (OpenGL)

    // Copy PBO -> Texture (Fast GPU-to-GPU copy)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw Fullscreen Quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1,  1);
    glTexCoord2f(0, 0); glVertex2f(-1,  1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    // force continuous rendering
    update();
}

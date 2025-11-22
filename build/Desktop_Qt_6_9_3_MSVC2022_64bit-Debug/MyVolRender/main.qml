import QtQuick
import QtQuick.Controls
import MyVolRender 1.0

Window {
    width: 800; height: 600
    visible: true
    color: "white"
    title: "3D Interactive Volume"

    VolumeItem {
        id: renderer
        anchors.fill: parent

        rotationX: 0.0
        rotationY: 0.0
        opecity: 0.0
    }

    MouseArea {
        anchors.fill: parent

        property real lastX: 0
        property real lastY: 0

        onPressed: (mouse) => {
            lastX = mouse.x
            lastY = mouse.y
        }

        onPositionChanged: (mouse) => {
            var deltaX = mouse.x - lastX
            var deltaY = mouse.y - lastY

            // sensitivity factor (0.01)
            renderer.rotationY += deltaX * 0.01
            renderer.rotationX += deltaY * 0.01

            lastX = mouse.x
            lastY = mouse.y
        }
    }

    Slider {
        id: opacitySlider
        from: 0.0
        to: 1.0
        value: 0.0

        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 20

        onValueChanged:
            renderer.opecity = value
        z: 1
    }

    Text {
        anchors.bottom: parent.bottom
        anchors.margins: 20
        anchors.horizontalCenter: parent.horizontalCenter
        text: "Click & Drag to Rotate (3D)"
        color: "white"
        font.pixelSize: 16
    }
}

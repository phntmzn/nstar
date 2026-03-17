import Cocoa

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var windowController: MainWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let controller = MainWindowController()
        windowController = controller
        controller.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

